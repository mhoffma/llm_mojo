"""Probe: is a spin barrier among a persistent team of threads cheaper than
starting a parallel region for every operation?

Profiling (`--profile`, PLAN.md M6) showed every matmul call paying ~80 us
whatever its size: the cost of a parallel region (`parallelize`), which
wakes the worker threads and waits for all of them. A decode step runs ~50
of them. llama.cpp avoids this by starting its threads once per step and
having them meet at a barrier between operations.

This compares, for S steps of work split across all threads:

  region    `parallelize` once per step (what the model does now)
  barrier   one `parallelize` whose workers loop over the steps and meet at
            a spin barrier after each (a shared arrival counter and a
            generation number: the last thread to arrive resets the counter
            and advances the generation; the others spin on it with PAUSE)

for three kinds of step: empty (pure overhead), tiny (~1 us of work per
thread), and matmul-sized (a ~0.3-1.2 MB read, like one int4 matmul of a
decode step, 48 per token).

Run with the environment the model uses:
    MODULAR_THREAD_BUSY_WAIT_US=0 uv run mojo run research/test_barrier.mojo

Results on the i7-1160G7, 8 threads, busy-wait 0, 2026-09-28 (us per step,
median of 3):

                         region   barrier
  empty step              22.8      1.2
  tiny step (~1 us)       23.6      0.4
  0.3 MB step             22.7      1.5
  1.2 MB step             30.1      7.5

(The buffers here fit in cache, so the reads are faster than the model's
streaming reads; the overhead is the point.) A parallel region costs ~23 us
of pure overhead; a spin barrier ~1 us. No deadlock: all 8 tasks of one
`parallelize` run at once, which a barrier needs. In the model, profiling
suggested ~80 us per region (threads sleep deeper between real work), so
running a decode step as one region with barriers between operations should
save roughly 1-4 ms of a 7-10 ms step.
"""

from std.atomic import Atomic
from std.memory.alloc import unsafe_alloc
from std.runtime import parallelism_level
from std.sys.intrinsics import llvm_intrinsic
from std.time import perf_counter_ns
from max.algorithm import parallelize

comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime FPtr = Pointer[Float32, MutUntrackedOrigin]
comptime W = 16


@always_inline
def cpu_pause():
    """PAUSE: tells the CPU this is a spin-wait loop (saves power, and frees
    the core for its hyperthread sibling)."""
    llvm_intrinsic["llvm.x86.sse2.pause", NoneType]()


struct SpinBarrier(ImplicitlyCopyable):
    """A reusable barrier for n threads, in shared memory.

    Each thread reads the generation, then adds itself to the arrival count.
    The last to arrive resets the count and advances the generation; the
    others spin until the generation changes. Resetting the count before
    advancing the generation makes it safe to reuse immediately.
    """

    var count: I64Ptr
    var gen: I64Ptr
    var n: Int

    def __init__(out self, n: Int):
        self.count = unsafe_alloc[Int64](8)  # a cache line each, apart
        self.gen = unsafe_alloc[Int64](8)
        Atomic[Int64].store(self.count, 0)
        Atomic[Int64].store(self.gen, 0)
        self.n = n

    @always_inline
    def wait(self):
        var g = Atomic[Int64].load(self.gen)
        if Atomic[Int64].fetch_add(self.count, 1) == Int64(self.n - 1):
            Atomic[Int64].store(self.count, 0)
            Atomic[Int64].store(self.gen, g + 1)
        else:
            while Atomic[Int64].load(self.gen) == g:
                cpu_pause()

    def free(self):
        self.count.unsafe_free()
        self.gen.unsafe_free()


@always_inline
def work(data: FPtr, begin: Int, end: Int) -> Float32:
    """Reads data[begin:end] (the stand-in for a matmul's weight reads)."""
    var s = SIMD[DType.float32, W](0)
    for i in range(begin, end, W):
        s += data.unsafe_load[width=W](i)
    return s.reduce_add()


def run_regions(data: FPtr, steps: Int, floats: Int, nt: Int, sink: FPtr) -> Float64:
    """Returns microseconds per step."""
    var t0 = perf_counter_ns()
    for step in range(steps):
        var off = (step % 4) * floats  # cycle through 4 buffers, like 4 matmuls

        def part(t: Int) {imm}:
            var chunk = floats // nt
            sink[unsafe_offset=t] += work(data, off + t * chunk, off + (t + 1) * chunk)

        parallelize(part, nt)
    return Float64(perf_counter_ns() - t0) / 1e3 / Float64(steps)


def run_barrier(data: FPtr, steps: Int, floats: Int, nt: Int, sink: FPtr) -> Float64:
    var bar = SpinBarrier(nt)
    var t0 = perf_counter_ns()

    def worker(t: Int) {imm}:
        var chunk = floats // nt
        for step in range(steps):
            var off = (step % 4) * floats
            sink[unsafe_offset=t] += work(data, off + t * chunk, off + (t + 1) * chunk)
            bar.wait()

    parallelize(worker, nt)
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(steps)
    bar.free()
    return us


def main():
    var nt = parallelism_level()
    print("threads:", nt)
    # 4 buffers of up to 1.2 MB-equivalent each (0.3M floats = 1.2 MB).
    var maxf = 300_000
    var data = unsafe_alloc[Float32](4 * maxf + 64)
    for i in range(4 * maxf + 64):
        data[unsafe_offset=i] = Float32(i % 7) * 0.001
    var sink = unsafe_alloc[Float32](64)
    for i in range(64):
        sink[unsafe_offset=i] = 0

    print("microseconds per step (median of 3):   region | barrier")
    for c in [(0, 2000, "empty step"), (16 * 8 * 16, 2000, "tiny step (~1 us)"), (75_000, 480, "0.3 MB step"), (300_000, 480, "1.2 MB step")]:
        var floats = c[0] // (nt * W) * (nt * W)
        var r = List[Float64]()
        var b = List[Float64]()
        for _ in range(3):
            r.append(run_regions(data, c[1], floats, nt, sink))
            b.append(run_barrier(data, c[1], floats, nt, sink))
        sort(r)
        sort(b)
        print("  ", c[2], ":", r[1], "|", b[1])

    if sink[unsafe_offset=0] == 12345:
        print(sink[unsafe_offset=0])
    data.unsafe_free()
    sink.unsafe_free()
