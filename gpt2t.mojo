"""GPT-2 124M inference on CPU, generic over the weight storage format.

Same program as gpt2.mojo, restructured so the big weight matrices can be
stored in different formats (see tensor.mojo). The model is `Model[W]` for a
format `W`; `main` picks W from --dtype and calls the generic `run[W]`, so
each format gets its own compiled copy of the whole forward pass.

Usage:
    mojo run gpt2t.mojo [--dtype FMT] [-m DIR] [-n TOKENS] [-t TEMP] [-k TOPK]
                        [-s SEED] [-v] "prompt"

    --dtype FMT  weight format: f32 (default), f16, bf16
    -m DIR       directory with the Hugging Face GPT-2 files (default: gpt2)
    -n TOKENS    number of tokens to generate (default: 64)
    -t TEMP      sampling temperature; 0 = greedy (default: 0.8)
    -k TOPK      sample from the TOPK most likely tokens; 0 = all (default: 40)
    -s SEED      random seed (default: 1337)
    -v           print prompt token ids and the top-5 next-token logits

Which tensors use the format: the four matrices in each layer (qkv, attention
projection, MLP up and down) and wte, the token embedding that doubles as the
output head. They are ~99% of the bytes read per token. LayerNorm parameters,
biases, and the position embedding wpe stay float32.
"""

from std.memory.alloc import unsafe_alloc
from std.sys import argv
from std.time import perf_counter_ns
from std.math import exp, log
from std.io import FileDescriptor
from std.os import SEEK_END, SEEK_SET

from tensor import FPtr, NW, F32V, WeightMatrix, DenseMatrix
from max.algorithm import parallelize
from kernels import (
    matmul,
    layernorm,
    attention,
    lm_head,
    lm_head_rows,
    MAX_PARTS,
)
from tokenizer import Tokenizer, parse_uint, read_file_bytes

# GPT-2 small hyperparameters.
comptime N_LAYER = 12
comptime N_HEAD = 12
comptime C = 768
comptime HS = C // N_HEAD
comptime MAX_T = 1024
comptime V = 50257
comptime EOT = 50256

# Per-layer weight matrices, in the order stored in Model.mats.
comptime QKV = 0
comptime PROJ = 1
comptime FC = 2
comptime FCPROJ = 3
comptime N_MATS = 4
comptime MAT_NAMES = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]
comptime MAT_ROWS = [C, C, C, 4 * C]
comptime MAT_COLS = [3 * C, C, 4 * C, C]

# Per-layer float32 vectors, in the order stored in Model.vecs.
comptime LN1_W = 0
comptime LN1_B = 1
comptime QKV_B = 2
comptime PROJ_B = 3
comptime LN2_W = 4
comptime LN2_B = 5
comptime FC_B = 6
comptime FCPROJ_B = 7
comptime N_VECS = 8
comptime VEC_NAMES = [
    "ln_1.weight",
    "ln_1.bias",
    "attn.c_attn.bias",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "mlp.c_fc.bias",
    "mlp.c_proj.bias",
]
comptime VEC_LENS = [C, C, 3 * C, C, C, C, 4 * C, C]
comptime LAYER_VEC_FLOATS = 13 * C  # sum of VEC_LENS


# ===----------------------------------------------------------------------=== #
# Model
# ===----------------------------------------------------------------------=== #


def tensor(header: String, params: FPtr, name: String) raises -> FPtr:
    """Finds a tensor's data_offsets in the safetensors JSON header."""
    var k = header.find('"' + name + '"')
    if k < 0:
        raise Error("tensor not found: " + name)
    var i = header.find('"data_offsets":[', k) + 16
    var off = parse_uint(header.as_bytes(), i)
    return params.unsafe_offset(off // 4)


def read_safetensors(path: String, mut header: String) raises -> FPtr:
    """Reads a float32 safetensors file; returns its payload, and sets
    `header` to its JSON header.

    `mut` passes header by reference, so the assignment is seen by the caller.
    """
    var f = open(path, "r")
    var hb = f.read_bytes(8)
    var hlen = 0
    for i in range(8):
        hlen |= Int(hb[i]) << (8 * i)
    header = String(from_utf8_lossy=Span(f.read_bytes(hlen)))
    var payload = Int(f.seek(0, SEEK_END)) - 8 - hlen
    _ = f.seek(8 + hlen, SEEK_SET)
    var params = unsafe_alloc[Float32](payload // 4)
    var done = 0
    while done < payload:
        var n = f.read(
            Span(
                unsafe_ptr=params.unsafe_offset(done // 4),
                length=(payload - done) // 4,
            )
        )
        if n <= 0 or n % 4 != 0:
            raise Error("short read on " + path)
        done += n
    f.close()
    return params


struct Model[W: WeightMatrix](Movable):
    """GPT-2 with its large matrices stored in format W."""

    var wte: Self.W  # [V, C] token embedding, also the output head
    var mats: List[Self.W]  # N_LAYER * N_MATS layer matrices
    var small: FPtr  # all float32 parameters, in one buffer:
    var vecs: List[FPtr]  #   N_LAYER * N_VECS pointers into small
    var wpe: FPtr  #   [MAX_T, C] position embedding
    var lnf_w: FPtr
    var lnf_b: FPtr
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
        """Loads float32 weights, converts the large ones to W, and frees
        the float32 copy."""
        var header = String()
        var params = read_safetensors(path, header)

        self.wte = Self.W.from_f32(tensor(header, params, "wte.weight"), V, C)
        self.mats = List[Self.W](capacity=N_LAYER * N_MATS)
        for l in range(N_LAYER):
            comptime for m in range(N_MATS):
                comptime name = MAT_NAMES[m]
                comptime rows = MAT_ROWS[m]
                comptime cols = MAT_COLS[m]
                var src = tensor(header, params, "h." + String(l) + "." + name)
                self.mats.append(Self.W.from_f32(src, rows, cols))

        # Copy the small float32 tensors out, so the payload can be freed.
        self.small = unsafe_alloc[Float32](
            N_LAYER * LAYER_VEC_FLOATS + MAX_T * C + 2 * C
        )
        var off = 0
        self.vecs = List[FPtr](capacity=N_LAYER * N_VECS)
        for l in range(N_LAYER):
            comptime for s in range(N_VECS):
                comptime name = VEC_NAMES[s]
                comptime n = VEC_LENS[s]
                var src = tensor(header, params, "h." + String(l) + "." + name)
                self.vecs.append(copy_into(self.small, off, src, n))
        self.wpe = copy_into(
            self.small, off, tensor(header, params, "wpe.weight"), MAX_T * C
        )
        self.lnf_w = copy_into(
            self.small, off, tensor(header, params, "ln_f.weight"), C
        )
        self.lnf_b = copy_into(
            self.small, off, tensor(header, params, "ln_f.bias"), C
        )
        params.unsafe_free()

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
        self.wte.free()
        for m in self.mats:
            m.free()
        self.small.unsafe_free()
        self.x.unsafe_free()
        self.xn.unsafe_free()
        self.qkv.unsafe_free()
        self.att.unsafe_free()
        self.fc.unsafe_free()
        self.logits.unsafe_free()
        self.kcache.unsafe_free()
        self.vcache.unsafe_free()
        self.scratch.unsafe_free()

    def weight_bytes(self) -> Int:
        """Bytes of all parameters, as stored."""
        var n = self.wte.nbytes()
        for m in self.mats:
            n += m.nbytes()
        return n + (N_LAYER * LAYER_VEC_FLOATS + MAX_T * C + 2 * C) * 4

    def forward(self, tokens: List[Int], pos0: Int) -> FPtr:
        """Runs tokens at positions pos0.. and returns the last token's logits.
        """
        self.blocks(tokens, pos0)
        var last = self.x.unsafe_offset((len(tokens) - 1) * C)
        layernorm[C](self.xn, last, self.lnf_w, self.lnf_b, 1)
        lm_head(self.logits, self.xn, self.wte, V, C)
        return self.logits

    def forward_all(self, tokens: List[Int], logits: FPtr):
        """Runs tokens from position 0 and writes logits for every position
        into logits[T, V]. Row t predicts tokens[t + 1]."""
        var T = len(tokens)
        self.blocks(tokens, 0)
        layernorm[C](self.xn, self.x, self.lnf_w, self.lnf_b, T)
        lm_head_rows(logits, self.xn, self.wte, T, V, C)

    def blocks(self, tokens: List[Int], pos0: Int):
        """Embeds tokens at positions pos0.. and runs all transformer blocks,
        leaving the result in self.x and filling the KV cache."""
        var T = len(tokens)
        var x = self.x
        var xn = self.xn
        var qkv = self.qkv
        for t in range(T):
            var p = self.wpe.unsafe_offset((pos0 + t) * C)
            for i in range(0, C, NW):
                x.unsafe_store(
                    t * C + i,
                    self.wte.load[NW](tokens[t], i)
                    + p.unsafe_load[width=NW](i),
                )

        for l in range(N_LAYER):
            var m = self.mats.unsafe_ptr().unsafe_offset(l * N_MATS)
            var v = self.vecs.unsafe_ptr().unsafe_offset(l * N_VECS)
            var kc = self.kcache.unsafe_offset(l * MAX_T * C)
            var vc = self.vcache.unsafe_offset(l * MAX_T * C)

            layernorm[C](
                xn, x, v[unsafe_offset=LN1_W], v[unsafe_offset=LN1_B], T
            )
            matmul(
                qkv,
                xn,
                m[unsafe_offset=QKV],
                v[unsafe_offset=QKV_B],
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
            attention[N_HEAD, HS](self.att, qkv, kc, vc, T, pos0)
            matmul[RESID=True](
                x,
                self.att,
                m[unsafe_offset=PROJ],
                v[unsafe_offset=PROJ_B],
                T,
                C,
                C,
                self.scratch,
            )

            layernorm[C](
                xn, x, v[unsafe_offset=LN2_W], v[unsafe_offset=LN2_B], T
            )
            matmul[GELU=True](
                self.fc,
                xn,
                m[unsafe_offset=FC],
                v[unsafe_offset=FC_B],
                T,
                C,
                4 * C,
                self.scratch,
            )
            matmul[RESID=True](
                x,
                self.fc,
                m[unsafe_offset=FCPROJ],
                v[unsafe_offset=FCPROJ_B],
                T,
                4 * C,
                C,
                self.scratch,
            )


def copy_into(dst: FPtr, mut off: Int, src: FPtr, n: Int) -> FPtr:
    """Copies n floats to dst at off, advances off, and returns the copy."""
    var p = dst.unsafe_offset(off)
    for i in range(n):
        p[unsafe_offset=i] = src[unsafe_offset=i]
    off += n
    return p


# ===----------------------------------------------------------------------=== #
# Sampling
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


# ===----------------------------------------------------------------------=== #
# Main
# ===----------------------------------------------------------------------=== #


struct Args(Movable):
    var dtype: String
    var dir: String
    var steps: Int
    var temperature: Float32
    var k: Int
    var seed: Int
    var verbose: Bool
    var prompt: String
    var ppl: String  # evaluation text file; empty = generate instead
    var compare: Bool  # with ppl: also compare against float32

    def __init__(out self) raises:
        self.dtype = "f32"
        self.dir = "gpt2"
        self.steps = 64
        self.temperature = 0.8
        self.k = 40
        self.seed = 1337
        self.verbose = False
        self.prompt = "The meaning of life is"
        self.ppl = ""
        self.compare = False
        var args = argv()
        var a = 1
        while a < len(args):
            var arg = String(args[a])
            if arg == "-v":
                self.verbose = True
            elif arg == "--compare":
                self.compare = True
            elif arg.startswith("-") and a + 1 < len(args):
                var val = String(args[a + 1])
                if arg == "--dtype":
                    self.dtype = val
                elif arg == "--ppl":
                    self.ppl = val
                elif arg == "-m":
                    self.dir = val
                elif arg == "-n":
                    self.steps = atol(val)
                elif arg == "-t":
                    self.temperature = Float32(atof(val))
                elif arg == "-k":
                    self.k = atol(val)
                elif arg == "-s":
                    self.seed = atol(val)
                else:
                    raise Error("unknown option " + arg)
                a += 1
            else:
                self.prompt = arg
            a += 1


def generate[W: WeightMatrix](args: Args) raises:
    """Loads the model with weights in format W and generates text."""
    var t_load = perf_counter_ns()
    var model = Model[W](args.dir + "/model.safetensors")
    var tok = Tokenizer(args.dir)
    var load_ms = Float64(perf_counter_ns() - t_load) / 1e6

    var tokens = tok.encode(args.prompt)
    if len(tokens) == 0:
        tokens.append(EOT)
    var steps = min(args.steps, MAX_T - len(tokens))
    if args.verbose:
        print("prompt tokens:", tokens)

    var t0 = perf_counter_ns()
    var logits = model.forward(tokens, 0)
    var prefill_ns = perf_counter_ns() - t0
    if args.verbose:
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
    out.write_bytes(Span(args.prompt.as_bytes()))
    var rng = Rng(args.seed)
    var pos = len(tokens)
    var generated = 0
    var t1 = perf_counter_ns()
    for _ in range(steps):
        var next = sample(logits, args.temperature, args.k, rng)
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
        "dtype",
        W.NAME,
        "|",
        model.weight_bytes() // (1024 * 1024),
        "MB | load",
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


# ===----------------------------------------------------------------------=== #
# Evaluation
# ===----------------------------------------------------------------------=== #

comptime F32 = DenseMatrix[DType.float32]
comptime EVAL_WINDOW = MAX_T
comptime EVAL_STRIDE = 512

# Per-position results, one array of each per window row.
comptime NLL = 0  # -log p(target) under the model being evaluated
comptime NLL_BASE = 1  # the same under float32 (compare mode)
comptime AGREE = 2  # 1 if both models' top-1 predictions are equal
comptime KL = 3  # KL(float32 || model), in nats
comptime DIFF_MAX = 4  # largest |logit difference| over the vocabulary
comptime DIFF_SUM = 5  # sum of |logit differences| over the vocabulary
comptime N_STATS = 6


@always_inline
def log_sum_exp(row: FPtr) -> Float64:
    """Returns log(sum(exp(row[v]))) over the vocabulary, computed stably."""
    var mx = row[unsafe_offset=0]
    for v in range(1, V):
        mx = max(mx, row[unsafe_offset=v])
    var s = Float64(0)
    for v in range(V):
        s += Float64(exp(row[unsafe_offset=v] - mx))
    return Float64(mx) + log(s)


def score_rows[
    COMPARE: Bool
](
    stats: Pointer[Float64, MutUntrackedOrigin],
    logits: FPtr,
    base: FPtr,
    targets: List[Int],
    r0: Int,
    r1: Int,
):
    """Fills stats[k * MAX_T + r] for window rows r0..r1-1.

    Row r's logits predict targets[r]. Rows are independent, so they are
    scored in parallel.
    """
    var tp = targets.unsafe_ptr()

    def row(i: Int) {imm}:
        var r = r0 + i
        var lm = logits.unsafe_offset(r * V)
        var target = tp[unsafe_offset=r]
        var lse = log_sum_exp(lm)
        stats[unsafe_offset = NLL * MAX_T + r] = lse - Float64(
            lm[unsafe_offset=target]
        )
        comptime if COMPARE:
            var lb = base.unsafe_offset(r * V)
            var lse_b = log_sum_exp(lb)
            var kl = Float64(0)
            var dmax = Float64(0)
            var dsum = Float64(0)
            var best = 0
            var best_b = 0
            for v in range(V):
                var m = lm[unsafe_offset=v]
                var b = lb[unsafe_offset=v]
                var lpb = Float64(b) - lse_b
                kl += exp(lpb) * (lpb - (Float64(m) - lse))
                var d = abs(Float64(m) - Float64(b))
                dmax = max(dmax, d)
                dsum += d
                if m > lm[unsafe_offset=best]:
                    best = v
                if b > lb[unsafe_offset=best_b]:
                    best_b = v
            stats[unsafe_offset = NLL_BASE * MAX_T + r] = lse_b - Float64(
                lb[unsafe_offset=target]
            )
            stats[unsafe_offset = AGREE * MAX_T + r] = Float64(
                Int(best == best_b)
            )
            stats[unsafe_offset = KL * MAX_T + r] = kl
            stats[unsafe_offset = DIFF_MAX * MAX_T + r] = dmax
            stats[unsafe_offset = DIFF_SUM * MAX_T + r] = dsum

    parallelize(row, r1 - r0)


def eval_loop[
    W: WeightMatrix, B: WeightMatrix, COMPARE: Bool
](model: Model[W], base: Model[B], ids: List[Int], name: String) raises:
    """Scores every token of ids after the first, in sliding windows.

    With COMPARE, also runs `base` (float32) on the same windows and compares
    the two models' predictions position by position.

    Windows are EVAL_WINDOW tokens long and start every EVAL_STRIDE tokens,
    and each token is scored once, in the first window where it isn't
    already scored. So apart from the first window, every scored token
    has at least EVAL_WINDOW - EVAL_STRIDE tokens of context.
    """
    var logits = unsafe_alloc[Float32](EVAL_WINDOW * V)
    var base_logits = unsafe_alloc[Float32](EVAL_WINDOW * V) if COMPARE else logits
    var stats = unsafe_alloc[Float64](N_STATS * MAX_T)
    var totals = List[Float64](length=N_STATS, fill=0)
    var diff_max = Float64(0)
    var scored = 0
    var windows = 0
    var n = len(ids)
    var scored_until = 0  # ids[1 .. scored_until] have been scored
    var begin = 0
    var t0 = perf_counter_ns()
    while True:
        var end = min(begin + EVAL_WINDOW, n)
        var window = List[Int](capacity=end - begin)
        var targets = List[Int](capacity=end - begin)
        for i in range(begin, end):
            window.append(ids[i])
            targets.append(ids[i + 1] if i + 1 < n else 0)
        model.forward_all(window, logits)
        comptime if COMPARE:
            base.forward_all(window, base_logits)
        # Row r predicts ids[begin + r + 1]; score only targets not yet seen.
        var r0 = max(0, scored_until - begin)
        var r1 = end - begin - 1
        score_rows[COMPARE](stats, logits, base_logits, targets, r0, r1)
        for r in range(r0, r1):
            for k in range(N_STATS):
                totals[k] += stats[unsafe_offset = k * MAX_T + r]
            diff_max = max(diff_max, stats[unsafe_offset = DIFF_MAX * MAX_T + r])
        scored += r1 - r0
        scored_until = end - 1
        windows += 1
        if end == n:
            break
        begin += EVAL_STRIDE
    var secs = Float64(perf_counter_ns() - t0) / 1e9

    var cnt = Float64(scored)
    var ppl = exp(totals[NLL] / cnt)
    print(
        name, ":", scored, "tokens scored in", windows, "windows of",
        EVAL_WINDOW, "(stride", EVAL_STRIDE, ") in", secs, "s",
    )
    print("  ", W.NAME, "perplexity", ppl, " mean NLL", totals[NLL] / cnt, "nats")
    comptime if COMPARE:
        var ppl_b = exp(totals[NLL_BASE] / cnt)
        print("   f32 perplexity", ppl_b, " change", (ppl / ppl_b - 1) * 100, "%")
        print("   top-1 agreement with f32:", totals[AGREE] / cnt * 100, "%")
        print("   mean KL(f32 || ", W.NAME, "):", totals[KL] / cnt, "nats")
        print(
            "   logit |diff| vs f32: mean", totals[DIFF_SUM] / (cnt * V),
            " max", diff_max,
        )
        base_logits.unsafe_free()
    logits.unsafe_free()
    stats.unsafe_free()


def evaluate[W: WeightMatrix](args: Args) raises:
    """--ppl: perplexity of the text in args.ppl, and with --compare, how far
    format W's predictions are from float32's."""
    var tok = Tokenizer(args.dir)
    var text = String(from_utf8_lossy=Span(read_file_bytes(args.ppl)))
    var ids = tok.encode(text)
    if len(ids) < 2:
        raise Error("need at least 2 tokens in " + args.ppl)
    var path = args.dir + "/model.safetensors"
    var model = Model[W](path)
    if args.compare:
        var base = Model[F32](path)
        eval_loop[W, F32, True](model, base, ids, args.ppl)
    else:
        # Without COMPARE the baseline is never used; pass model itself.
        eval_loop[W, W, False](model, model, ids, args.ppl)


def run[W: WeightMatrix](args: Args) raises:
    if args.ppl.byte_length() > 0:
        evaluate[W](args)
    else:
        generate[W](args)


def main() raises:
    var args = Args()
    # Each branch instantiates the whole program for one format, at compile
    # time. Adding a format is one more line here.
    if args.dtype == "f32":
        run[DenseMatrix[DType.float32]](args)
    elif args.dtype == "f16":
        run[DenseMatrix[DType.float16]](args)
    elif args.dtype == "bf16":
        run[DenseMatrix[DType.bfloat16]](args)
    else:
        raise Error(
            "unknown --dtype " + args.dtype + " (supported: f32, f16, bf16)"
        )
