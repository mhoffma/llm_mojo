"""GPT-2 124M inference on CPU, in Mojo.

Reads the Hugging Face files directly (model.safetensors, vocab.json,
merges.txt), tokenizes the prompt with GPT-2's byte-level BPE, runs the
transformer with a KV cache, and streams sampled tokens to stdout.

Usage:
    mojo run gpt2.mojo [-m DIR] [-n TOKENS] [-t TEMP] [-k TOPK] [-s SEED] [-v] "prompt"

    -m DIR     directory with the Hugging Face GPT-2 files (default: gpt2)
    -n TOKENS  number of tokens to generate (default: 64)
    -t TEMP    sampling temperature; 0 = greedy (default: 0.8)
    -k TOPK    sample from the TOPK most likely tokens; 0 = all (default: 40)
    -s SEED    random seed (default: 1337)
    -v         print prompt token ids and the top-5 next-token logits

Build once with `mojo build -o gpt2_bin gpt2.mojo` for the fastest startup.
Decoding is memory-bandwidth bound and runs ~60 parallel regions per token; on
small laptop CPUs, idle workers busy-waiting between regions steal cycles from
the ones doing work, so `MODULAR_THREAD_BUSY_WAIT_US=0` can speed it up a lot.
"""

from std.memory.alloc import unsafe_alloc
from std.sys import argv, simd_width_of
from std.time import perf_counter_ns
from std.collections import Dict
from std.math import exp, tanh, sqrt
from std.io import FileDescriptor
from std.os import SEEK_END, SEEK_SET
from max.algorithm import parallelize
from std.runtime import parallelism_level

comptime FPtr = Pointer[Float32, MutUntrackedOrigin]
comptime NW = simd_width_of[DType.float32]()
comptime F32V = SIMD[DType.float32, NW]

# GPT-2 small hyperparameters.
comptime N_LAYER = 12
comptime N_HEAD = 12
comptime C = 768
comptime HS = C // N_HEAD
comptime MAX_T = 1024
comptime V = 50257
comptime EOT = 50256
comptime MAX_PARTS = 64  # max threads used by gemv

# Per-layer weight slots, in the order stored in Model.layer_w.
comptime LN1_W = 0
comptime LN1_B = 1
comptime QKV_W = 2
comptime QKV_B = 3
comptime PROJ_W = 4
comptime PROJ_B = 5
comptime LN2_W = 6
comptime LN2_B = 7
comptime FC_W = 8
comptime FC_B = 9
comptime FCPROJ_W = 10
comptime FCPROJ_B = 11
comptime N_SLOTS = 12

comptime LAYER_NAMES = [
    "ln_1.weight",
    "ln_1.bias",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
]


# ===----------------------------------------------------------------------=== #
# Kernels
# ===----------------------------------------------------------------------=== #


@always_inline
def store_out[RESID: Bool, GELU: Bool](o: FPtr, off: Int, var r: F32V):
    """Applies the optional GELU, adds the residual, and stores."""
    comptime if GELU:
        comptime s = Float32(0.7978845608028654)  # sqrt(2/pi)
        r = 0.5 * r * (1 + tanh(s * (r + 0.044715 * r * r * r)))
    comptime if RESID:
        r += o.unsafe_load[width=NW](off)
    o.unsafe_store(off, r)


@always_inline
def mm_tile[
    TM: Int, NV: Int, RESID: Bool, GELU: Bool
](out_: FPtr, x: FPtr, w: FPtr, b: FPtr, t0: Int, j0: Int, IN: Int, OUT: Int,):
    """Computes a TM x (NV*NW) tile of out = x @ w + b, held in registers."""
    var acc = Array[F32V, length=TM * NV](fill=F32V(0))
    var wv = Array[F32V, length=NV](fill=F32V(0))
    for i in range(IN):
        var wrow = w.unsafe_offset(i * OUT + j0)
        comptime for v in range(NV):
            wv[v] = wrow.unsafe_load[width=NW](v * NW)
        comptime for m in range(TM):
            var xm = F32V(x[unsafe_offset=(t0 + m) * IN + i])
            comptime for v in range(NV):
                acc[m * NV + v] = xm.fma(wv[v], acc[m * NV + v])
    comptime for m in range(TM):
        var orow = out_.unsafe_offset((t0 + m) * OUT + j0)
        comptime for v in range(NV):
            var r = acc[m * NV + v] + b.unsafe_load[width=NW](j0 + v * NW)
            store_out[RESID, GELU](orow, v * NW, r)


def matmul[
    RESID: Bool = False, GELU: Bool = False
](
    out_: FPtr,
    x: FPtr,
    w: FPtr,
    b: FPtr,
    T: Int,
    IN: Int,
    OUT: Int,
    scratch: FPtr,
):
    """Computes out[T, OUT] (+)= act(x[T, IN] @ w[IN, OUT] + b[OUT]).

    Weights use the GPT-2 Conv1D layout [IN, OUT], so each input row
    contributes a contiguous SIMD-friendly slice of the output. For several
    rows (prefill), work is split across threads by column blocks, and rows are
    processed TM at a time so each weight load is reused TM times. For a single
    row (decode), see gemv.
    """
    if T == 1:
        gemv[RESID, GELU](out_, x, w, b, IN, OUT, scratch)
        return
    comptime NV = 2
    comptime TN = NV * NW
    comptime TM = 8
    debug_assert(OUT % TN == 0, "OUT must be a multiple of the tile width")

    def block(blk: Int) {imm}:
        var j0 = blk * TN
        var t = 0
        while t + TM <= T:
            mm_tile[TM, NV, RESID, GELU](out_, x, w, b, t, j0, IN, OUT)
            t += TM
        while t < T:
            mm_tile[1, NV, RESID, GELU](out_, x, w, b, t, j0, IN, OUT)
            t += 1

    parallelize(block, OUT // TN)


def gemv[
    RESID: Bool, GELU: Bool
](out_: FPtr, x: FPtr, w: FPtr, b: FPtr, IN: Int, OUT: Int, scratch: FPtr):
    """Single-row matmul, split over IN so each thread streams whole rows.

    Decoding is bound by memory bandwidth: every weight is read once per token.
    Giving each thread a contiguous band of rows keeps reads sequential; the
    per-thread partial sums (in scratch) are then reduced.
    """
    var nparts = min(parallelism_level(), MAX_PARTS)
    var rows = (IN + nparts - 1) // nparts

    def part(pi: Int) {imm}:
        var acc = scratch.unsafe_offset(pi * OUT)
        for j in range(0, OUT, NW):
            acc.unsafe_store(j, F32V(0))
        var i = pi * rows
        var end = min(IN, i + rows)
        while i + 4 <= end:
            var x0 = F32V(x[unsafe_offset=i])
            var x1 = F32V(x[unsafe_offset=i + 1])
            var x2 = F32V(x[unsafe_offset=i + 2])
            var x3 = F32V(x[unsafe_offset=i + 3])
            var w0 = w.unsafe_offset(i * OUT)
            var w1 = w0.unsafe_offset(OUT)
            var w2 = w1.unsafe_offset(OUT)
            var w3 = w2.unsafe_offset(OUT)
            for j in range(0, OUT, NW):
                var s = acc.unsafe_load[width=NW](j)
                s = x0.fma(w0.unsafe_load[width=NW](j), s)
                s = x1.fma(w1.unsafe_load[width=NW](j), s)
                s = x2.fma(w2.unsafe_load[width=NW](j), s)
                s = x3.fma(w3.unsafe_load[width=NW](j), s)
                acc.unsafe_store(j, s)
            i += 4
        while i < end:
            var xi = F32V(x[unsafe_offset=i])
            var wi = w.unsafe_offset(i * OUT)
            for j in range(0, OUT, NW):
                acc.unsafe_store(
                    j,
                    xi.fma(
                        wi.unsafe_load[width=NW](j),
                        acc.unsafe_load[width=NW](j),
                    ),
                )
            i += 1

    parallelize(part, nparts)
    for j in range(0, OUT, NW):
        var r = b.unsafe_load[width=NW](j)
        for pi in range(nparts):
            r += scratch.unsafe_load[width=NW](pi * OUT + j)
        store_out[RESID, GELU](out_, j, r)


def layernorm(out_: FPtr, x: FPtr, w: FPtr, b: FPtr, T: Int):
    for t in range(T):
        var xr = x.unsafe_offset(t * C)
        var o = out_.unsafe_offset(t * C)
        var s = F32V(0)
        for i in range(0, C, NW):
            s += xr.unsafe_load[width=NW](i)
        var mean = s.reduce_add() / C
        var sq = F32V(0)
        for i in range(0, C, NW):
            var d = xr.unsafe_load[width=NW](i) - mean
            sq += d * d
        var rstd = 1 / sqrt(sq.reduce_add() / C + 1e-5)
        for i in range(0, C, NW):
            var n = (xr.unsafe_load[width=NW](i) - mean) * rstd
            o.unsafe_store(
                i, n * w.unsafe_load[width=NW](i) + b.unsafe_load[width=NW](i)
            )


def attention(out_: FPtr, qkv: FPtr, kc: FPtr, vc: FPtr, T: Int, pos0: Int):
    """Causal multi-head attention for T new tokens at positions pos0.. .

    Queries come from qkv; keys and values come from the layer's KV cache,
    which already holds positions 0 .. pos0+T-1.
    """
    var scale = 1 / sqrt(Float32(HS))

    def head_query(idx: Int) {imm}:
        var h = idx % N_HEAD
        var t = idx // N_HEAD
        var npos = pos0 + t + 1
        var q = qkv.unsafe_offset(t * 3 * C + h * HS)
        var scores = unsafe_alloc[Float32](npos)
        var mx = Float32.MIN
        for s in range(npos):
            var k = kc.unsafe_offset(s * C + h * HS)
            var d = F32V(0)
            comptime for i in range(0, HS, NW):
                d += q.unsafe_load[width=NW](i) * k.unsafe_load[width=NW](i)
            var sc = d.reduce_add() * scale
            scores[unsafe_offset=s] = sc
            mx = max(mx, sc)
        var total = Float32(0)
        for s in range(npos):
            var e = exp(scores[unsafe_offset=s] - mx)
            scores[unsafe_offset=s] = e
            total += e
        var o = out_.unsafe_offset(t * C + h * HS)
        var acc = Array[F32V, length=HS // NW](fill=F32V(0))
        for s in range(npos):
            var p = F32V(scores[unsafe_offset=s] / total)
            var v = vc.unsafe_offset(s * C + h * HS)
            comptime for i in range(HS // NW):
                acc[i] = p.fma(v.unsafe_load[width=NW](i * NW), acc[i])
        comptime for i in range(HS // NW):
            o.unsafe_store(i * NW, acc[i])
        scores.unsafe_free()

    # Waking the worker threads costs more than a short-context decode step's
    # attention, so only go parallel when there is enough work.
    if T * (pos0 + T) < 256:
        for i in range(N_HEAD * T):
            head_query(i)
    else:
        parallelize(head_query, N_HEAD * T)


def lm_head(logits: FPtr, h: FPtr, wte: FPtr):
    """Computes logits[V] = h[C] @ wte[V, C]^T (the output head is tied to wte).
    """
    comptime CHUNK = 512
    var nchunks = (V + CHUNK - 1) // CHUNK

    def chunk(ci: Int) {imm}:
        var end = min(V, (ci + 1) * CHUNK)
        for v in range(ci * CHUNK, end):
            var row = wte.unsafe_offset(v * C)
            var d = F32V(0)
            for i in range(0, C, NW):
                d = h.unsafe_load[width=NW](i).fma(
                    row.unsafe_load[width=NW](i), d
                )
            logits[unsafe_offset=v] = d.reduce_add()

    parallelize(chunk, nchunks)


# ===----------------------------------------------------------------------=== #
# Model
# ===----------------------------------------------------------------------=== #


def read_file_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def parse_uint(b: Span[UInt8, _], mut i: Int) -> Int:
    var n = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        n = n * 10 + Int(b[i] - 48)
        i += 1
    return n


def tensor(header: String, params: FPtr, name: String) raises -> FPtr:
    """Finds a tensor's data_offsets in the safetensors JSON header."""
    var k = header.find('"' + name + '"')
    if k < 0:
        raise Error("tensor not found: " + name)
    var i = header.find('"data_offsets":[', k) + 16
    var off = parse_uint(header.as_bytes(), i)
    return params.unsafe_offset(off // 4)


struct Model(Movable):
    var params: FPtr  # the whole safetensors payload, as float32
    var wte: FPtr
    var wpe: FPtr
    var lnf_w: FPtr
    var lnf_b: FPtr
    var layer_w: List[FPtr]  # N_LAYER * N_SLOTS pointers into params
    # Activations, sized for MAX_T tokens.
    var x: FPtr
    var xn: FPtr
    var qkv: FPtr
    var att: FPtr
    var fc: FPtr
    var logits: FPtr
    var kcache: FPtr  # [N_LAYER, MAX_T, C]
    var vcache: FPtr
    var scratch: FPtr  # gemv partial sums

    def __init__(out self, path: String) raises:
        var f = open(path, "r")
        var hb = f.read_bytes(8)
        var hlen = 0
        for i in range(8):
            hlen |= Int(hb[i]) << (8 * i)
        var header = String(from_utf8_lossy=Span(f.read_bytes(hlen)))
        var payload = Int(f.seek(0, SEEK_END)) - 8 - hlen
        _ = f.seek(8 + hlen, SEEK_SET)
        self.params = unsafe_alloc[Float32](payload // 4)
        var done = 0
        while done < payload:
            var n = f.read(
                Span(
                    unsafe_ptr=self.params.unsafe_offset(done // 4),
                    length=(payload - done) // 4,
                )
            )
            if n <= 0 or n % 4 != 0:
                raise Error("short read on " + path)
            done += n
        f.close()

        self.wte = tensor(header, self.params, "wte.weight")
        self.wpe = tensor(header, self.params, "wpe.weight")
        self.lnf_w = tensor(header, self.params, "ln_f.weight")
        self.lnf_b = tensor(header, self.params, "ln_f.bias")
        self.layer_w = List[FPtr](capacity=N_LAYER * N_SLOTS)
        for l in range(N_LAYER):
            comptime for s in range(N_SLOTS):
                comptime name = LAYER_NAMES[s]
                self.layer_w.append(
                    tensor(header, self.params, "h." + String(l) + "." + name)
                )

        self.x = unsafe_alloc[Float32](MAX_T * C)
        self.xn = unsafe_alloc[Float32](MAX_T * C)
        self.qkv = unsafe_alloc[Float32](MAX_T * 3 * C)
        self.att = unsafe_alloc[Float32](MAX_T * C)
        self.fc = unsafe_alloc[Float32](MAX_T * 4 * C)
        self.logits = unsafe_alloc[Float32](V)
        self.kcache = unsafe_alloc[Float32](N_LAYER * MAX_T * C)
        self.vcache = unsafe_alloc[Float32](N_LAYER * MAX_T * C)
        self.scratch = unsafe_alloc[Float32](MAX_PARTS * 4 * C)

    def __deinit__(deinit self):
        self.params.unsafe_free()
        self.x.unsafe_free()
        self.xn.unsafe_free()
        self.qkv.unsafe_free()
        self.att.unsafe_free()
        self.fc.unsafe_free()
        self.logits.unsafe_free()
        self.kcache.unsafe_free()
        self.vcache.unsafe_free()
        self.scratch.unsafe_free()

    def forward(self, tokens: List[Int], pos0: Int) -> FPtr:
        """Runs tokens at positions pos0.. and returns the last token's logits.
        """
        var T = len(tokens)
        var x = self.x
        var xn = self.xn
        var qkv = self.qkv
        for t in range(T):
            var e = self.wte.unsafe_offset(tokens[t] * C)
            var p = self.wpe.unsafe_offset((pos0 + t) * C)
            for i in range(0, C, NW):
                x.unsafe_store(
                    t * C + i,
                    e.unsafe_load[width=NW](i) + p.unsafe_load[width=NW](i),
                )

        for l in range(N_LAYER):
            var w = self.layer_w.unsafe_ptr().unsafe_offset(l * N_SLOTS)
            var kc = self.kcache.unsafe_offset(l * MAX_T * C)
            var vc = self.vcache.unsafe_offset(l * MAX_T * C)

            layernorm(xn, x, w[unsafe_offset=LN1_W], w[unsafe_offset=LN1_B], T)
            matmul(
                qkv,
                xn,
                w[unsafe_offset=QKV_W],
                w[unsafe_offset=QKV_B],
                T,
                C,
                3 * C,
                self.scratch,
            )
            for t in range(T):
                var row = qkv.unsafe_offset(t * 3 * C)
                var dst = (pos0 + t) * C
                for i in range(0, C, NW):
                    kc.unsafe_store(dst + i, row.unsafe_load[width=NW](C + i))
                    vc.unsafe_store(
                        dst + i, row.unsafe_load[width=NW](2 * C + i)
                    )
            attention(self.att, qkv, kc, vc, T, pos0)
            matmul[RESID=True](
                x,
                self.att,
                w[unsafe_offset=PROJ_W],
                w[unsafe_offset=PROJ_B],
                T,
                C,
                C,
                self.scratch,
            )

            layernorm(xn, x, w[unsafe_offset=LN2_W], w[unsafe_offset=LN2_B], T)
            matmul[GELU=True](
                self.fc,
                xn,
                w[unsafe_offset=FC_W],
                w[unsafe_offset=FC_B],
                T,
                C,
                4 * C,
                self.scratch,
            )
            matmul[RESID=True](
                x,
                self.fc,
                w[unsafe_offset=FCPROJ_W],
                w[unsafe_offset=FCPROJ_B],
                T,
                4 * C,
                C,
                self.scratch,
            )

        var last = x.unsafe_offset((T - 1) * C)
        layernorm(xn, last, self.lnf_w, self.lnf_b, 1)
        lm_head(self.logits, xn, self.wte)
        return self.logits


# ===----------------------------------------------------------------------=== #
# Tokenizer (GPT-2 byte-level BPE)
# ===----------------------------------------------------------------------=== #


def byte_to_codepoint() -> List[Int]:
    """GPT-2's reversible byte -> printable-unicode mapping (bytes_to_unicode).
    """
    var table = List[Int](length=256, fill=0)
    var n = 0
    for b in range(256):
        var printable = (
            (b >= 33 and b <= 126)
            or (b >= 161 and b <= 172)
            or (b >= 174 and b <= 255)
        )
        if printable:
            table[b] = b
        else:
            table[b] = 256 + n
            n += 1
    return table^


def utf8_len(lead: UInt8) -> Int:
    if lead < 0x80:
        return 1
    if lead < 0xE0:
        return 2
    if lead < 0xF0:
        return 3
    return 4


def decode_codepoint(b: Span[UInt8, _], i: Int) -> Int:
    var n = utf8_len(b[i])
    if n == 1 or i + n > len(b):
        return Int(b[i])
    var cp = Int(b[i]) & (0x7F >> n)
    for k in range(1, n):
        cp = (cp << 6) | (Int(b[i + k]) & 0x3F)
    return cp


comptime LETTER = 0
comptime DIGIT = 1
comptime SPACE = 2
comptime OTHER = 3


def char_class(cp: Int) -> Int:
    """Approximates the \\p{L} / \\p{N} / \\s classes of GPT-2's pre-tokenizer
    regex for the scripts most text uses."""
    if (cp >= 97 and cp <= 122) or (cp >= 65 and cp <= 90):
        return LETTER
    if cp >= 48 and cp <= 57:
        return DIGIT
    if cp == 32 or (cp >= 9 and cp <= 13) or cp == 0x85 or cp == 0xA0:
        return SPACE
    if (cp >= 0x2000 and cp <= 0x200A) or cp == 0x2028 or cp == 0x2029:
        return SPACE
    if cp == 0x202F or cp == 0x205F or cp == 0x3000:
        return SPACE
    if cp < 0xAA:
        return OTHER
    if cp == 0xAA or cp == 0xB5 or cp == 0xBA:
        return LETTER
    if cp >= 0xC0 and cp <= 0x24F and cp != 0xD7 and cp != 0xF7:
        return LETTER
    if cp >= 0x370 and cp <= 0x1FFF:
        return LETTER  # Greek, Cyrillic, Hebrew, Arabic, Indic, ...
    if cp >= 0x3040 and cp <= 0x9FFF:
        return LETTER  # kana, CJK
    if cp >= 0xAC00 and cp <= 0xD7AF:
        return LETTER  # Hangul
    return OTHER


struct Tokenizer(Movable):
    var token_bytes: List[List[UInt8]]  # id -> raw bytes
    var byte_id: List[Int]  # byte -> single-byte token id
    var merges: Dict[Int, Int]  # (a << 16 | b) -> (rank << 16 | merged id)

    def __init__(out self, dir: String) raises:
        var to_cp = byte_to_codepoint()
        var from_cp = List[Int](length=324, fill=0)
        for b in range(256):
            from_cp[to_cp[b]] = b

        # vocab.json: {"<token>": id, ...}, token text in bytes_to_unicode form.
        var js = read_file_bytes(dir + "/vocab.json")
        var ids = Dict[String, Int]()
        self.token_bytes = List[List[UInt8]](length=V, fill=List[UInt8]())
        var i = 0
        while i < len(js):
            if Int(js[i]) != ord('"'):
                i += 1
                continue
            i += 1
            var key = String()
            var raw = List[UInt8]()
            while Int(js[i]) != ord('"'):
                var cp: Int
                if Int(js[i]) == ord("\\"):
                    var esc = Int(js[i + 1])
                    if esc == ord("u"):
                        var hex = String(
                            from_utf8_lossy=Span(js)[i + 2 : i + 6]
                        )
                        cp = atol(hex, base=16)
                        i += 6
                    else:
                        cp = esc  # \" \\ \/
                        if esc == ord("n"):
                            cp = 10
                        elif esc == ord("t"):
                            cp = 9
                        elif esc == ord("r"):
                            cp = 13
                        i += 2
                else:
                    cp = decode_codepoint(Span(js), i)
                    i += utf8_len(js[i])
                key += chr(cp)
                raw.append(UInt8(from_cp[cp]))
            i += 1
            while Int(js[i]) == ord(":") or Int(js[i]) == ord(" "):
                i += 1
            var id = parse_uint(Span(js), i)
            ids[key] = id
            self.token_bytes[id] = raw^

        self.byte_id = List[Int](length=256, fill=0)
        for b in range(256):
            self.byte_id[b] = ids[chr(to_cp[b])]

        # merges.txt: one "a b" pair per line, in priority order.
        var text = String(
            from_utf8_lossy=Span(read_file_bytes(dir + "/merges.txt"))
        )
        self.merges = Dict[Int, Int]()
        var rank = 0
        for line in text.split("\n"):
            if line.startswith("#version") or line.byte_length() == 0:
                continue
            var parts = line.split(" ")
            var a = ids[String(parts[0])]
            var b = ids[String(parts[1])]
            var m = ids[String(parts[0]) + String(parts[1])]
            self.merges[(a << 16) | b] = (rank << 16) | m
            rank += 1

    def bpe(self, mut toks: List[Int]):
        """Repeatedly merges the lowest-ranked adjacent pair."""
        while len(toks) > 1:
            var best_rank = Int.MAX
            var best_pair = -1
            var best_id = -1
            for i in range(len(toks) - 1):
                var key = (toks[i] << 16) | toks[i + 1]
                var r = self.merges.find(key)
                if r and (r.value() >> 16) < best_rank:
                    best_rank = r.value() >> 16
                    best_id = r.value() & 0xFFFF
                    best_pair = key
            if best_pair < 0:
                return
            var merged = List[Int](capacity=len(toks))
            var i = 0
            while i < len(toks):
                if (
                    i + 1 < len(toks)
                    and ((toks[i] << 16) | toks[i + 1]) == best_pair
                ):
                    merged.append(best_id)
                    i += 2
                else:
                    merged.append(toks[i])
                    i += 1
            toks = merged^

    def encode(self, text: String) -> List[Int]:
        var b = text.as_bytes()
        var out = List[Int]()
        var s = 0
        for e in pretokenize(b):
            var toks = List[Int](capacity=e - s)
            for k in range(s, e):
                toks.append(self.byte_id[Int(b[k])])
            self.bpe(toks)
            out.extend(toks^)
            s = e
        return out^


def class_at(b: Span[UInt8, _], k: Int) -> Int:
    return char_class(decode_codepoint(b, k))


def pretokenize(b: Span[UInt8, _]) -> List[Int]:
    """Splits text into chunks the way GPT-2's regex does.

    The regex is:

        's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

    Returns the end offset of each chunk. BPE merges never cross chunks.
    """
    var n = len(b)
    var ends = List[Int]()
    var i = 0
    while i < n:
        var c0 = Int(b[i])
        # Contractions: 's 't 're 've 'm 'll 'd
        if c0 == ord("'") and i + 1 < n:
            var c1 = Int(b[i + 1])
            if (
                c1 == ord("s")
                or c1 == ord("t")
                or c1 == ord("m")
                or c1 == ord("d")
            ):
                i += 2
                ends.append(i)
                continue
            if i + 2 < n:
                var c2 = Int(b[i + 2])
                if (
                    (c1 == ord("r") and c2 == ord("e"))
                    or (c1 == ord("v") and c2 == ord("e"))
                    or (c1 == ord("l") and c2 == ord("l"))
                ):
                    i += 3
                    ends.append(i)
                    continue
        # Optional leading space, then a run of letters, digits, or symbols.
        var j = i
        if c0 == ord(" ") and i + 1 < n and class_at(b, i + 1) != SPACE:
            j = i + 1
        var cls = class_at(b, j)
        if cls != SPACE:
            var k = j + utf8_len(b[j])
            while k < n and class_at(b, k) == cls:
                k += utf8_len(b[k])
            ends.append(k)
            i = k
            continue
        # Whitespace: \s+(?!\S) leaves the last space to prefix the next word.
        var k = i
        var last = i
        var count = 0
        while k < n and class_at(b, k) == SPACE:
            last = k
            k += utf8_len(b[k])
            count += 1
        i = last if (k < n and count > 1) else k
        ends.append(i)
    return ends^


# ===----------------------------------------------------------------------=== #
# Sampling and main
# ===----------------------------------------------------------------------=== #


struct Rng(Movable):
    var state: UInt64

    def __init__(out self, seed: Int):
        self.state = UInt64(seed) * 0x9E3779B97F4A7C15 + 1

    def uniform(mut self) -> Float32:
        """Advances a xorshift64* generator, returning a float in [0, 1)."""
        self.state ^= self.state >> 12
        self.state ^= self.state << 25
        self.state ^= self.state >> 27
        var r = (self.state * 0x2545F4914F6CDD1D) >> 40
        return Float32(r) / Float32(1 << 24)


def argmax(logits: FPtr) -> Int:
    var best = 0
    for v in range(1, V):
        if logits[unsafe_offset=v] > logits[unsafe_offset=best]:
            best = v
    return best


def top_k(logits: FPtr, k: Int) -> List[Int]:
    """Ids of the k largest logits, in descending order."""
    var ids = List[Int](capacity=k + 1)
    for v in range(V):
        var lv = logits[unsafe_offset=v]
        if len(ids) == k and lv <= logits[unsafe_offset=ids[k - 1]]:
            continue
        var pos = len(ids)
        ids.append(v)
        while pos > 0 and logits[unsafe_offset=ids[pos - 1]] < lv:
            ids[pos] = ids[pos - 1]
            pos -= 1
        ids[pos] = v
        if len(ids) > k:
            _ = ids.pop()
    return ids^


def sample(logits: FPtr, temperature: Float32, k: Int, mut rng: Rng) -> Int:
    if temperature <= 0:
        return argmax(logits)
    var cand = top_k(logits, k if k > 0 else V)
    var mx = logits[unsafe_offset=cand[0]]
    var probs = List[Float32](capacity=len(cand))
    var total = Float32(0)
    for id in cand:
        var p = exp((logits[unsafe_offset=id] - mx) / temperature)
        probs.append(p)
        total += p
    var r = rng.uniform() * total
    for i in range(len(cand)):
        r -= probs[i]
        if r <= 0:
            return cand[i]
    return cand[len(cand) - 1]


def main() raises:
    var dir = String("gpt2")
    var steps = 64
    var temperature = Float32(0.8)
    var k = 40
    var seed = 1337
    var verbose = False
    var prompt = String("The meaning of life is")
    var args = argv()
    var a = 1
    while a < len(args):
        var arg = String(args[a])
        if arg == "-v":
            verbose = True
        elif arg.startswith("-") and a + 1 < len(args):
            var val = String(args[a + 1])
            if arg == "-m":
                dir = val
            elif arg == "-n":
                steps = atol(val)
            elif arg == "-t":
                temperature = Float32(atof(val))
            elif arg == "-k":
                k = atol(val)
            elif arg == "-s":
                seed = atol(val)
            else:
                raise Error("unknown option " + arg)
            a += 1
        else:
            prompt = arg
        a += 1

    var t_load = perf_counter_ns()
    var model = Model(dir + "/model.safetensors")
    var tok = Tokenizer(dir)
    var load_ms = Float64(perf_counter_ns() - t_load) / 1e6

    var tokens = tok.encode(prompt)
    if len(tokens) == 0:
        tokens.append(EOT)
    steps = min(steps, MAX_T - len(tokens))
    if verbose:
        print("prompt tokens:", tokens)

    var t0 = perf_counter_ns()
    var logits = model.forward(tokens, 0)
    var prefill_ns = perf_counter_ns() - t0
    if verbose:
        var best = top_k(logits, 5)
        print("top-5 next tokens:")
        for id in best:
            print(
                "  ",
                id,
                repr(String(from_utf8_lossy=Span(tok.token_bytes[id]))),
                logits[unsafe_offset=id],
            )

    var out = FileDescriptor(1)
    out.write_bytes(Span(prompt.as_bytes()))
    var rng = Rng(seed)
    var pos = len(tokens)
    var generated = 0
    var t1 = perf_counter_ns()
    for _ in range(steps):
        var next = sample(logits, temperature, k, rng)
        if next == EOT:
            break
        out.write_bytes(Span(tok.token_bytes[next]))
        generated += 1
        if generated == steps:
            break
        var one = List[Int]()
        one.append(next)
        logits = model.forward(one, pos)
        pos += 1
    var decode_ns = perf_counter_ns() - t1

    var prefill_ms = Float64(prefill_ns) / 1e6
    var decode_ms = Float64(decode_ns) / 1e6
    print("\n---", file=FileDescriptor(2))
    print(
        "load",
        Int(load_ms),
        "ms | prompt",
        len(tokens),
        "tokens in",
        Int(prefill_ms),
        "ms (",
        Int(Float64(len(tokens)) * 1000 / prefill_ms),
        "tok/s ) | generated",
        generated,
        "tokens in",
        Int(decode_ms),
        "ms (",
        Int(Float64(generated) * 1000 / decode_ms) if decode_ms > 0 else 0,
        "tok/s )",
        file=FileDescriptor(2),
    )
