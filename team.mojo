"""A persistent team of threads that meet at spin barriers.

Starting a parallel region (`parallelize`) wakes the worker threads and waits
for all of them: ~23-80 us each time, whatever the work (research/
test_barrier.mojo, PLAN.md M6). A decode step used ~50 of them. Instead, the
model starts one region per decode step and its threads go through every
operation together, each doing its share and then waiting at a barrier
(~1 us) before the next operation, as llama.cpp does.

All `nt` tasks of one `parallelize(worker, nt)` must run at the same time for
a barrier to work: nt = parallelism_level(), one task per runtime worker
(checked by the probe: no deadlock).
"""

from std.atomic import Atomic
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.sys.intrinsics import llvm_intrinsic

comptime I64Ptr = Pointer[Int64, MutUntrackedOrigin]

comptime SPINS = 1000
"""PAUSEs a waiting thread spends before it starts yielding its CPU
(sched_yield) between checks. Pure spinning is fastest when every thread
has a core to itself, but on a busy machine a descheduled team member makes
the others spin for a whole scheduler time slice; yielding lets it run."""


@always_inline
def cpu_pause():
    """PAUSE: marks a spin-wait loop (saves power, frees the core's other
    hyperthread)."""
    llvm_intrinsic["llvm.x86.sse2.pause", NoneType]()


struct Team(ImplicitlyCopyable):
    """nt threads and a reusable spin barrier for them, in shared memory.

    wait(): each thread reads the generation, then adds itself to the arrival
    count. The last to arrive resets the count and advances the generation;
    the others spin until the generation changes. Resetting the count before
    advancing the generation makes the barrier reusable at once. The count
    and generation are 64 bytes apart (separate cache lines).
    """

    var count: I64Ptr
    var gen: I64Ptr
    var nt: Int

    def __init__(out self, nt: Int):
        self.count = unsafe_alloc[Int64](8)
        self.gen = unsafe_alloc[Int64](8)
        Atomic[Int64].store(self.count, 0)
        Atomic[Int64].store(self.gen, 0)
        self.nt = nt

    @always_inline
    def wait(self):
        var g = Atomic[Int64].load(self.gen)
        if Atomic[Int64].fetch_add(self.count, 1) == Int64(self.nt - 1):
            Atomic[Int64].store(self.count, 0)
            Atomic[Int64].store(self.gen, g + 1)
        else:
            var spins = 0
            while Atomic[Int64].load(self.gen) == g:
                if spins < SPINS:
                    cpu_pause()
                    spins += 1
                else:
                    _ = external_call["sched_yield", Int32]()

    def free(self):
        self.count.unsafe_free()
        self.gen.unsafe_free()


@always_inline
def split(n: Int, tid: Int, nt: Int) -> Tuple[Int, Int]:
    """Thread tid's share [begin, end) of n items split into nt contiguous
    ranges."""
    var per = (n + nt - 1) // nt
    return (min(n, tid * per), min(n, (tid + 1) * per))
