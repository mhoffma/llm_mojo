"""Unit test for int_attention.IntAttnKV.attend (integer attention).

Stores random keys and values, runs integer attention for several context
lengths and token counts (decode: T = 1; prefill: T > 1), and compares with
exact float64 attention on the same quantized keys, values and query (int8
keys and values with a scale per position and head, int16 query with a scale
per head). What remains is the integer softmax (Q15), the fixed-point score
rescaling, and the int16 weights; the error should be a small fraction of
the output's magnitude.

    uv run mojo run -I . tests/int_attention_check.mojo
"""

from std.math import exp, sqrt, round
from std.memory.alloc import unsafe_alloc
from std.random import randn_float64, seed

from int_attention import IntAttnKV

comptime NH = 12
comptime HS = 64
comptime C = NH * HS
comptime MAXT = 1024


def q8(x: Float64, s: Float64) -> Float64:
    return round(x / s)


def check(npos0: Int, T: Int, spread: Float64) -> Float64:
    """Returns max |int - exact| / max |exact| over the T tokens' outputs."""
    var kv = IntAttnKV.create(1, MAXT, NH, HS)
    var n = npos0 + T
    # Keys and values for all positions, kept in float for the reference.
    var kf = unsafe_alloc[Float32](n * C)
    var vf = unsafe_alloc[Float32](n * C)
    for i in range(n * C):
        kf[unsafe_offset=i] = Float32(randn_float64(0, spread))
        vf[unsafe_offset=i] = Float32(randn_float64(0, 1))
    for pos in range(n):
        kv.store(0, pos, kf.unsafe_offset(pos * C), vf.unsafe_offset(pos * C))
    # Queries: rows of a [T, 3C] qkv buffer (only the q part is read).
    var qkv = unsafe_alloc[Float32](T * 3 * C)
    for i in range(T * 3 * C):
        qkv[unsafe_offset=i] = Float32(randn_float64(0, spread))
    var out = unsafe_alloc[Float32](T * C)
    kv.attend[NH, HS](out, qkv, 0, T, npos0)

    var worst = Float64(0)
    var biggest = Float64(0)
    for t in range(T):
        var npos = npos0 + t + 1
        for h in range(NH):
            # The same quantization, in float64.
            var mq = Float64(0)
            for d in range(HS):
                mq = max(mq, abs(Float64(qkv[unsafe_offset = t * 3 * C + h * HS + d])))
            var sq = Float64(Float32(mq) / 32767)
            var sc = List[Float64](length=npos, fill=0)
            var m = Float64(-1e300)
            for s in range(npos):
                var mk = Float64(0)
                for d in range(HS):
                    mk = max(mk, abs(Float64(kf[unsafe_offset = s * C + h * HS + d])))
                var sk = Float64(Float32(mk) / 127)
                var dot = Float64(0)
                for d in range(HS):
                    var qi = round(Float64(qkv[unsafe_offset = t * 3 * C + h * HS + d] * (1 / Float32(sq))))
                    var ki = round(Float64(kf[unsafe_offset = s * C + h * HS + d] * (1 / Float32(sk))))
                    dot += qi * ki
                sc[s] = dot * sq * sk / sqrt(Float64(HS))
                m = max(m, sc[s])
            var z = Float64(0)
            for s in range(npos):
                z += exp(sc[s] - m)
            for d in range(HS):
                var want = Float64(0)
                for s in range(npos):
                    var mv = Float64(0)
                    for dd in range(HS):
                        mv = max(mv, abs(Float64(vf[unsafe_offset = s * C + h * HS + dd])))
                    var sv = Float64(Float32(mv) / 127)
                    var vi = round(Float64(vf[unsafe_offset = s * C + h * HS + d] * (1 / Float32(sv))))
                    want += exp(sc[s] - m) / z * vi * sv
                var got = Float64(out[unsafe_offset = t * C + h * HS + d])
                worst = max(worst, abs(got - want))
                biggest = max(biggest, abs(want))
    kv.free()
    kf.unsafe_free()
    vf.unsafe_free()
    qkv.unsafe_free()
    out.unsafe_free()
    return worst / biggest


def main():
    seed(5)
    var worst = Float64(0)
    print("integer attention vs exact float64 on the same quantized data:")
    for c in [(0, 1), (0, 5), (15, 1), (16, 1), (99, 1), (200, 7), (511, 1), (1000, 3)]:
        for spread in [0.5, 2.0]:
            var e = check(c[0], c[1], spread)
            worst = max(worst, e)
            print(
                "   positions before:", c[0], "tokens:", c[1], "spread:", spread,
                "-> max error / max |out|:", e,
            )
    print("worst:", worst, "PASS" if worst < 5e-3 else "FAIL")
