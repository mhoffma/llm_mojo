"""GPT-2 124M inference on CPU, generic over the weight storage format.

Same program as gpt2.mojo, restructured so the big weight matrices can be
stored in different formats (see tensor.mojo). The model is `Model[W, E]` for
a layer format `W` and head format `E`; `main` picks them from --dtype and
--head and calls the generic `run[W, E]`, so
each format gets its own compiled copy of the whole forward pass.

Usage:
    mojo run gpt2t.mojo [--dtype FMT] [-m DIR] [-n TOKENS] [-t TEMP] [-k TOPK]
                        [-s SEED] [-v] "prompt"

    --dtype FMT  weight format: f32 (default), f16, bf16, or quantized
                 int8-ch, int4-ch, int4-g128, int4-g64, int4-g32, each
                 also with a -sym suffix (see tensor.mojo QuantMatrix), and
                 -a16 variants (int8-ch-a16, int4-g32-a16, ...) that compute
                 with int16 activations in integers
    -m DIR       directory with the Hugging Face GPT-2 files (default: gpt2)
    -n TOKENS    number of tokens to generate (default: 64)
    -t TEMP      sampling temperature; 0 = greedy (default: 0.8)
    -k TOPK      sample from the TOPK most likely tokens; 0 = all (default: 40)
    -s SEED      random seed (default: 1337)
    -v           print prompt token ids and the top-5 next-token logits
    --head FMT   with an int --dtype: the tied embedding/output head's format,
                 int8 (default; int8, per vocabulary row) or same (as --dtype)
    --kv FMT     KV cache format: auto (default: f32/f16/bf16 for those
                 weights, int8 for quantized ones), f32, f16, bf16, int16,
                 int8
    --attention  float (default) or int: attention in integers (int8 cache
                 in VNNI layouts, integer softmax; int_attention.mojo)
    --profile    print time per operation (per decode and prompt token) at
                 the end
    --threads N  threads that decode each token together (default: half
                 the runtime's parallelism level, i.e. one per core with
                 hyperthreading: 4 on a 4-core/8-thread CPU)
    --gguf FILE  use the weights in a llama.cpp GGUF file of GPT-2 124M
                 (Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, F16, F32 tensors), as
                 stored; overrides --dtype. The tokenizer still comes from -m.
    --ppl FILE   measure perplexity on a text file instead of generating
    --compare    with --ppl: also run float32 and compare predictions

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

from tensor import FPtr, NW, F32V, I16Ptr, WeightMatrix, DenseMatrix, QuantMatrix
from max.algorithm import parallelize
from kernels import (
    linear,
    head,
    layernorm,
    attention,
    linear_team,
    head_team,
    MAX_PARTS,
)
from team import Team
from std.runtime import parallelism_level
from kvcache import KVCache, DenseKV, QuantKV
from int_attention import IntAttnKV
from gguf import GGUFFile, GGUFMatrix, BPtr, GGML_F32, type_name
from tokenizer import Tokenizer, parse_uint, read_file_bytes
from serialize import ByteWriter, ByteReader

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

# Profiling (--profile): time per operation, for prompt and decode tokens.
comptime P_EMBED = 0
comptime P_LN = 1  # both LayerNorms and the final one
comptime P_QKV = 2
comptime P_KVSTORE = 3
comptime P_ATTN = 4
comptime P_PROJ = 5
comptime P_FC = 6
comptime P_FCPROJ = 7
comptime P_HEAD = 8
comptime P_N = 9
comptime P_NAMES = [
    "embedding",
    "layernorm",
    "qkv matmul",
    "kv store",
    "attention",
    "attn proj matmul",
    "mlp up matmul",
    "mlp down matmul",
    "output head",
]


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


trait LanguageModel:
    """What the evaluation loop needs from a model."""

    def forward_all(self, tokens: List[Int], logits: FPtr):
        """Logits at every position of tokens (from position 0)."""
        ...


struct Model[
    W: WeightMatrix,
    E: WeightMatrix = W,
    KV: KVCache = DenseKV[DType.float32],
](LanguageModel, Movable):
    """GPT-2 with its layer matrices stored in format W, and its token
    embedding / output head (wte, lm) in format E, W by default.

    The output head is the tensor most sensitive to quantization, so the
    quantized formats keep it at int8 (see HEAD8), as llama.cpp keeps
    GPT-2's at 6-bit.

    Build one with `load_model` (Hugging Face safetensors, converted to W) or
    `load_gguf` (a llama.cpp GGUF file, used as stored).
    """

    var wte: Self.E  # [V, C] token embedding
    var lm: Self.E  # [V, C] output head: the same handle as wte when tied
    var tied: Bool
    var mats: List[Self.W]  # N_LAYER * N_MATS layer matrices
    var file: BPtr  # file buffer the matrices point into (GGUF), or unused
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
    var kv: Self.KV  # keys and values of past tokens, all layers
    var scratch: FPtr  # gemv partial sums
    var prof: Pointer[Int, MutUntrackedOrigin]  # ns per [phase][P_*]; phase 0 = decode, 1 = prompt
    var prof_tokens: Pointer[Int, MutUntrackedOrigin]  # tokens per phase
    var team: Team  # the threads that decode a token together (team.mojo)
    var team_i16: I16Ptr  # decode's quantized / reordered activations (2 x 4C)
    var team_f: FPtr  # their per-lane sums (4C / 16) and scale (2)

    def __init__(
        out self,
        wte: Self.E,
        lm: Self.E,
        tied: Bool,
        var mats: List[Self.W],
        small_src: List[FPtr],
        file: BPtr,
    ):
        """Takes the matrices, copies the small float32 tensors (listed in
        SMALL_ORDER) into one buffer, and allocates the activations."""
        self.wte = wte
        self.lm = lm
        self.tied = tied
        self.mats = mats^
        self.file = file
        self.small = unsafe_alloc[Float32](SMALL_FLOATS)
        var off = 0
        self.vecs = List[FPtr](capacity=N_LAYER * N_VECS)
        for l in range(N_LAYER):
            comptime for s in range(N_VECS):
                comptime n = VEC_LENS[s]
                self.vecs.append(
                    copy_into(self.small, off, small_src[l * N_VECS + s], n)
                )
        var k = N_LAYER * N_VECS
        self.wpe = copy_into(self.small, off, small_src[k], MAX_T * C)
        self.lnf_w = copy_into(self.small, off, small_src[k + 1], C)
        self.lnf_b = copy_into(self.small, off, small_src[k + 2], C)

        self.x = unsafe_alloc[Float32](MAX_T * C)
        self.xn = unsafe_alloc[Float32](MAX_T * C)
        self.qkv = unsafe_alloc[Float32](MAX_T * 3 * C)
        self.att = unsafe_alloc[Float32](MAX_T * C)
        self.fc = unsafe_alloc[Float32](MAX_T * 4 * C)
        self.logits = unsafe_alloc[Float32](V)
        self.kv = Self.KV.create(N_LAYER, MAX_T, N_HEAD, HS)
        self.scratch = unsafe_alloc[Float32](MAX_PARTS * 4 * C)
        self.prof = unsafe_alloc[Int](2 * P_N)
        # One team member per physical core: with 2 hyperthreads per core, a
        # waiting thread slows its sibling (measured: 4 threads decode ~40%
        # faster than 8 on the 4-core/8-thread i7-1160G7; PLAN.md M6).
        self.team = Team(max(1, parallelism_level() // 2))
        self.team_i16 = unsafe_alloc[Int16](2 * 4 * C)
        self.team_f = unsafe_alloc[Float32](4 * C // 16 + 2)
        self.prof_tokens = unsafe_alloc[Int](2)
        for i in range(2 * P_N):
            self.prof[unsafe_offset=i] = 0
        self.prof_tokens[unsafe_offset=0] = 0
        self.prof_tokens[unsafe_offset=1] = 0

    def __deinit__(deinit self):
        self.wte.free()
        if not self.tied:
            self.lm.free()
        for m in self.mats:
            m.free()
        self.file.unsafe_free()
        self.small.unsafe_free()
        self.x.unsafe_free()
        self.xn.unsafe_free()
        self.qkv.unsafe_free()
        self.att.unsafe_free()
        self.fc.unsafe_free()
        self.logits.unsafe_free()
        self.kv.free()
        self.scratch.unsafe_free()
        self.prof.unsafe_free()
        self.team.free()
        self.team_i16.unsafe_free()
        self.team_f.unsafe_free()
        self.prof_tokens.unsafe_free()

    def weight_bytes(self) -> Int:
        """Bytes of all parameters, as stored."""
        var n = self.wte.nbytes()
        if not self.tied:
            n += self.lm.nbytes()
        for m in self.mats:
            n += m.nbytes()
        return n + SMALL_FLOATS * 4

    @always_inline
    def tick(self, cat: Int, phase: Int, t0: Int) -> Int:
        """Adds the time since t0 to category cat; returns the current time.
        """
        var now = perf_counter_ns()
        self.prof[unsafe_offset = phase * P_N + cat] += Int(now - t0)
        return now

    def forward(self, tokens: List[Int], pos0: Int) -> FPtr:
        """Runs tokens at positions pos0.. and returns the last token's logits.
        One token (decoding) goes through `decode`, as one parallel region."""
        if len(tokens) == 1:
            return self.decode(tokens[0], pos0)
        var ph = 1
        self.blocks(tokens, pos0)
        var t = perf_counter_ns()
        var last = self.x.unsafe_offset((len(tokens) - 1) * C)
        layernorm[C](self.xn, last, self.lnf_w, self.lnf_b, 1)
        t = self.tick(P_LN, ph, t)
        head(self.logits, self.xn, self.lm, 1, V, C)
        _ = self.tick(P_HEAD, ph, t)
        return self.logits

    def set_threads(mut self, n: Int):
        """Uses a team of n threads for decoding (default: the runtime's
        parallelism level)."""
        self.team.free()
        self.team = Team(n)

    def decode(self, token: Int, pos: Int) -> FPtr:
        """One token at position pos through the whole model; returns its
        logits.

        The whole step is one parallel region: a team of threads goes
        through every operation together (linear_team, attend_one per head,
        head_team), meeting at a spin barrier after each, instead of starting
        a parallel region per operation (~50 per token, ~23-80 us each;
        PLAN.md M6). Thread 0 does the small serial pieces (embedding,
        LayerNorm, storing the token's key and value) and the profiling.
        The arithmetic is the same as the region-per-operation path.
        """
        self.prof_tokens[unsafe_offset=0] += 1
        var team = self.team
        var nt = team.nt
        var x = self.x
        var xn = self.xn
        var qkv = self.qkv
        var att = self.att
        var fc = self.fc
        var scratch = self.scratch
        var xq = self.team_i16
        var xp = self.team_i16.unsafe_offset(4 * C)
        var xs = self.team_f
        var sx = self.team_f.unsafe_offset(4 * C // 16)

        def worker(tid: Int) {imm}:
            var tm = perf_counter_ns()
            if tid == 0:
                var p = self.wpe.unsafe_offset(pos * C)
                self.wte.dequant_row(token, x)
                for i in range(0, C, NW):
                    x.unsafe_store(
                        i, x.unsafe_load[width=NW](i) + p.unsafe_load[width=NW](i)
                    )
                tm = self.tick(P_EMBED, 0, tm)
            for l in range(N_LAYER):
                var m = self.mats.unsafe_ptr().unsafe_offset(l * N_MATS)
                var v = self.vecs.unsafe_ptr().unsafe_offset(l * N_VECS)
                if tid == 0:
                    layernorm[C](
                        xn, x, v[unsafe_offset=LN1_W], v[unsafe_offset=LN1_B], 1
                    )
                team.wait()
                if tid == 0:
                    tm = self.tick(P_LN, 0, tm)
                linear_team(
                    tid, team, qkv, xn, m[unsafe_offset=QKV],
                    v[unsafe_offset=QKV_B], C, 3 * C, scratch, xq, xp, xs, sx,
                )
                if tid == 0:
                    tm = self.tick(P_QKV, 0, tm)
                    self.kv.store(
                        l, pos, qkv.unsafe_offset(C), qkv.unsafe_offset(2 * C)
                    )
                team.wait()
                if tid == 0:
                    tm = self.tick(P_KVSTORE, 0, tm)
                for h in range(tid, N_HEAD, nt):
                    self.kv.attend_one[N_HEAD, HS](att, qkv, l, 0, h, pos)
                team.wait()
                if tid == 0:
                    tm = self.tick(P_ATTN, 0, tm)
                linear_team[RESID=True](
                    tid, team, x, att, m[unsafe_offset=PROJ],
                    v[unsafe_offset=PROJ_B], C, C, scratch, xq, xp, xs, sx,
                )
                if tid == 0:
                    tm = self.tick(P_PROJ, 0, tm)
                    layernorm[C](
                        xn, x, v[unsafe_offset=LN2_W], v[unsafe_offset=LN2_B], 1
                    )
                team.wait()
                if tid == 0:
                    tm = self.tick(P_LN, 0, tm)
                linear_team[GELU=True](
                    tid, team, fc, xn, m[unsafe_offset=FC],
                    v[unsafe_offset=FC_B], C, 4 * C, scratch, xq, xp, xs, sx,
                )
                if tid == 0:
                    tm = self.tick(P_FC, 0, tm)
                linear_team[RESID=True](
                    tid, team, x, fc, m[unsafe_offset=FCPROJ],
                    v[unsafe_offset=FCPROJ_B], 4 * C, C, scratch, xq, xp, xs, sx,
                )
                if tid == 0:
                    tm = self.tick(P_FCPROJ, 0, tm)
            if tid == 0:
                layernorm[C](xn, x, self.lnf_w, self.lnf_b, 1)
            team.wait()
            if tid == 0:
                tm = self.tick(P_LN, 0, tm)
            head_team(tid, team, self.logits, xn, self.lm, V, C, xq, xp, xs, sx)
            if tid == 0:
                _ = self.tick(P_HEAD, 0, tm)

        parallelize(worker, nt)
        return self.logits

    def forward_all(self, tokens: List[Int], logits: FPtr):
        """Runs tokens from position 0 and writes logits for every position
        into logits[T, V]. Row t predicts tokens[t + 1]."""
        var T = len(tokens)
        var ph = 0 if T == 1 else 1
        self.blocks(tokens, 0)
        var t = perf_counter_ns()
        layernorm[C](self.xn, self.x, self.lnf_w, self.lnf_b, T)
        t = self.tick(P_LN, ph, t)
        head(logits, self.xn, self.lm, T, V, C)
        _ = self.tick(P_HEAD, ph, t)

    def print_profile(self):
        """Prints time per token by operation, for decode and prompt tokens
        (--profile)."""
        for ph in range(2):
            var n = self.prof_tokens[unsafe_offset=ph]
            if n == 0:
                continue
            var total = 0
            for c in range(P_N):
                total += self.prof[unsafe_offset = ph * P_N + c]
            print(
                "decode" if ph == 0 else "prompt", "(", n, "tokens):",
                Float64(total) / 1e6 / Float64(n), "ms per token",
                file=FileDescriptor(2),
            )
            comptime for c in range(P_N):
                comptime label = P_NAMES[c]
                var ns = self.prof[unsafe_offset = ph * P_N + c]
                print(
                    "   ", label, ":", Float64(ns) / 1e3 / Float64(n), "us  (",
                    Int(Float64(ns) * 100 / Float64(max(total, 1))), "% )",
                    file=FileDescriptor(2),
                )

    def blocks(self, tokens: List[Int], pos0: Int):
        """Embeds tokens at positions pos0.. and runs all transformer blocks,
        leaving the result in self.x and filling the KV cache."""
        var T = len(tokens)
        var ph = 0 if T == 1 else 1
        self.prof_tokens[unsafe_offset=ph] += T
        var x = self.x
        var xn = self.xn
        var qkv = self.qkv
        var tm = perf_counter_ns()
        for t in range(T):
            var p = self.wpe.unsafe_offset((pos0 + t) * C)
            var xt = x.unsafe_offset(t * C)
            self.wte.dequant_row(tokens[t], xt)
            for i in range(0, C, NW):
                xt.unsafe_store(
                    i, xt.unsafe_load[width=NW](i) + p.unsafe_load[width=NW](i)
                )
        tm = self.tick(P_EMBED, ph, tm)

        for l in range(N_LAYER):
            var m = self.mats.unsafe_ptr().unsafe_offset(l * N_MATS)
            var v = self.vecs.unsafe_ptr().unsafe_offset(l * N_VECS)

            layernorm[C](
                xn, x, v[unsafe_offset=LN1_W], v[unsafe_offset=LN1_B], T
            )
            tm = self.tick(P_LN, ph, tm)
            linear(
                qkv,
                xn,
                m[unsafe_offset=QKV],
                v[unsafe_offset=QKV_B],
                T,
                C,
                3 * C,
                self.scratch,
            )
            tm = self.tick(P_QKV, ph, tm)
            # Each new token's key and value (columns C.. and 2C.. of its qkv
            # row) go into the cache, then attention reads it.
            for t in range(T):
                var row = qkv.unsafe_offset(t * 3 * C)
                self.kv.store(
                    l, pos0 + t, row.unsafe_offset(C), row.unsafe_offset(2 * C)
                )
            tm = self.tick(P_KVSTORE, ph, tm)
            attention[N_HEAD, HS](self.att, qkv, self.kv, l, T, pos0)
            tm = self.tick(P_ATTN, ph, tm)
            linear[RESID=True](
                x,
                self.att,
                m[unsafe_offset=PROJ],
                v[unsafe_offset=PROJ_B],
                T,
                C,
                C,
                self.scratch,
            )
            tm = self.tick(P_PROJ, ph, tm)

            layernorm[C](
                xn, x, v[unsafe_offset=LN2_W], v[unsafe_offset=LN2_B], T
            )
            tm = self.tick(P_LN, ph, tm)
            linear[GELU=True](
                self.fc,
                xn,
                m[unsafe_offset=FC],
                v[unsafe_offset=FC_B],
                T,
                C,
                4 * C,
                self.scratch,
            )
            tm = self.tick(P_FC, ph, tm)
            linear[RESID=True](
                x,
                self.fc,
                m[unsafe_offset=FCPROJ],
                v[unsafe_offset=FCPROJ_B],
                T,
                4 * C,
                C,
                self.scratch,
            )
            tm = self.tick(P_FCPROJ, ph, tm)


# The small float32 tensors, in the order Model's constructor takes them:
# N_VECS per layer, then wpe, ln_f.weight, ln_f.bias.
comptime SMALL_FLOATS = N_LAYER * LAYER_VEC_FLOATS + MAX_T * C + 2 * C


def load_model[
    W: WeightMatrix, E: WeightMatrix = W, KV: KVCache = DenseKV[DType.float32]
](path: String) raises -> Model[W, E, KV]:
    """Loads Hugging Face float32 safetensors and converts the layer
    matrices to W and wte (the tied embedding and output head) to E."""
    var header = String()
    var params = read_safetensors(path, header)
    var wte = E.from_f32(
        tensor(header, params, "wte.weight"), V, C, reduce_rows=False
    )
    var mats = List[W](capacity=N_LAYER * N_MATS)
    var small = List[FPtr]()
    for l in range(N_LAYER):
        var prefix = "h." + String(l) + "."
        comptime for m in range(N_MATS):
            comptime name = MAT_NAMES[m]
            comptime rows = MAT_ROWS[m]
            comptime cols = MAT_COLS[m]
            var src = tensor(header, params, prefix + name)
            mats.append(W.from_f32(src, rows, cols, reduce_rows=True))
        comptime for s in range(N_VECS):
            comptime name = VEC_NAMES[s]
            small.append(tensor(header, params, prefix + name))
    small.append(tensor(header, params, "wpe.weight"))
    small.append(tensor(header, params, "ln_f.weight"))
    small.append(tensor(header, params, "ln_f.bias"))
    var model = Model[W, E, KV](
        wte, wte, True, mats^, small, unsafe_alloc[UInt8](1)
    )
    params.unsafe_free()  # the model copied what it keeps
    return model^


# Saved models (--save / --weights, M7): the matrices already converted to
# their formats, so loading is just reading (no quantization). Layout, in
# serialize.mojo's encoding:
#   SAVED_MAGIC, SAVED_VERSION, W.name(), E.name(),
#   wte (E.save), the N_LAYER * N_MATS layer matrices (W.save, in load_model's
#   order), SMALL_FLOATS, then the small float32 tensors as Model.small holds
#   them (SMALL_ORDER).
comptime SAVED_MAGIC = "llm_mojo GPT-2 124M weights"
comptime SAVED_VERSION = 1


def save_model[
    W: WeightMatrix, E: WeightMatrix, KV: KVCache
](model: Model[W, E, KV], path: String) raises:
    """Writes the model's weights, as converted, to path."""
    if not model.tied:
        raise Error("--save supports models with a tied output head only")
    var w = ByteWriter(path)
    w.string(SAVED_MAGIC)
    w.int(SAVED_VERSION)
    w.string(W.name())
    w.string(E.name())
    model.wte.save(w)
    for m in model.mats:
        m.save(w)
    w.int(SMALL_FLOATS)
    w.buffer(model.small, SMALL_FLOATS)
    w.close()
    print("saved", W.name(), "+ head", E.name(), "to", path, "(", w.written // 1000000, "MB )")


def check_magic(mut r: ByteReader, path: String) raises:
    """Raises unless the file starts with SAVED_MAGIC and SAVED_VERSION."""
    var magic = String(SAVED_MAGIC)
    if r.int() != magic.byte_length() or r.chars(magic.byte_length()) != magic:
        raise Error(path + " is not a saved llm_mojo model (see --save)")
    var version = r.int()
    if version != SAVED_VERSION:
        raise Error(path + ": unsupported version " + String(version))


def saved_formats(path: String) raises -> Tuple[String, String]:
    """The layer and head format names of a saved model file."""
    var r = ByteReader(path)
    check_magic(r, path)
    var layers = r.string()
    var head = r.string()
    r.close()
    return (layers, head)


def load_saved[
    W: WeightMatrix, E: WeightMatrix = W, KV: KVCache = DenseKV[DType.float32]
](path: String) raises -> Model[W, E, KV]:
    """Loads a model written by save_model; its formats must be W and E."""
    var r = ByteReader(path)
    check_magic(r, path)
    var layers = r.string()
    var head = r.string()
    if layers != W.name() or head != E.name():
        raise Error(
            path + " holds " + layers + " + head " + head + ", not "
            + W.name() + " + head " + E.name()
        )
    var wte = E.restore(r)
    var mats = List[W](capacity=N_LAYER * N_MATS)
    for _ in range(N_LAYER * N_MATS):
        mats.append(W.restore(r))
    if r.int() != SMALL_FLOATS:
        raise Error(path + ": unexpected size of the float32 tensors")
    var small = r.buffer[DType.float32](SMALL_FLOATS)
    r.close()
    # Pointers into small in Model's order; the constructor copies them.
    var small_src = List[FPtr]()
    var off = 0
    for _ in range(N_LAYER):
        comptime for s in range(N_VECS):
            comptime n = VEC_LENS[s]
            small_src.append(small.unsafe_offset(off))
            off += n
    small_src.append(small.unsafe_offset(off))  # wpe
    off += MAX_T * C
    small_src.append(small.unsafe_offset(off))  # ln_f weight
    small_src.append(small.unsafe_offset(off + C))  # ln_f bias
    var model = Model[W, E, KV](wte, wte, True, mats^, small_src, unsafe_alloc[UInt8](1))
    small.unsafe_free()
    return model^


# GGUF names for MAT_NAMES and VEC_NAMES (llama.cpp's GPT-2 conversion).
comptime GGUF_MAT_NAMES = [
    "attn_qkv.weight",
    "attn_output.weight",
    "ffn_up.weight",
    "ffn_down.weight",
]
comptime GGUF_VEC_NAMES = [
    "attn_norm.weight",
    "attn_norm.bias",
    "attn_qkv.bias",
    "attn_output.bias",
    "ffn_norm.weight",
    "ffn_norm.bias",
    "ffn_up.bias",
    "ffn_down.bias",
]


def gguf_f32(g: GGUFFile, name: String) raises -> FPtr:
    """A float32 tensor of a GGUF file, in place."""
    var m = g.matrix(name)
    if m.kind != GGML_F32:
        raise Error(name + " is " + type_name(m.kind) + ", expected F32")
    return m.data.unsafe_bitcast[Float32]()


def load_gguf[
    KV: KVCache = DenseKV[DType.float32]
](path: String) raises -> Model[GGUFMatrix, GGUFMatrix, KV]:
    """Loads a llama.cpp GGUF file of GPT-2 124M, using its matrices as
    stored (quantized), without converting them."""
    var g = GGUFFile(path)
    var wte = g.matrix("token_embd.weight")
    var tied = not g.has("output.weight")
    var lm = wte if tied else g.matrix("output.weight")
    var mats = List[GGUFMatrix](capacity=N_LAYER * N_MATS)
    var small = List[FPtr]()
    for l in range(N_LAYER):
        var prefix = "blk." + String(l) + "."
        comptime for m in range(N_MATS):
            comptime name = GGUF_MAT_NAMES[m]
            comptime rows = MAT_COLS[m]  # GGUF stores [OUT, IN]
            comptime cols = MAT_ROWS[m]
            var mat = g.matrix(prefix + name)
            if mat.rows != rows or mat.cols != cols:
                raise Error(prefix + name + " has an unexpected shape")
            mats.append(mat)
        comptime for s in range(N_VECS):
            comptime name = GGUF_VEC_NAMES[s]
            small.append(gguf_f32(g, prefix + name))
    small.append(gguf_f32(g, "position_embd.weight"))
    small.append(gguf_f32(g, "output_norm.weight"))
    small.append(gguf_f32(g, "output_norm.bias"))
    if wte.rows != V or wte.cols != C:
        raise Error("token_embd.weight has an unexpected shape")
    # The model takes over the file buffer: the matrices point into it.
    return Model[GGUFMatrix, GGUFMatrix, KV](
        wte, lm, tied, mats^, small, g^.release()
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
    var gguf: String  # GGUF weights file; overrides --dtype
    var head: String  # output head format for int formats: int8 or same
    var kv: String  # KV cache format, or "auto" for the weights' default
    var attention: String  # "float", or "int" (integer attention, int8 cache)
    var profile: Bool  # print time per operation at the end
    var threads: Int  # decode team size; 0 = the runtime's parallelism level
    var save: String  # write the converted weights here (M7)
    var weights: String  # load weights saved with --save instead

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
        self.gguf = ""
        self.head = "int8"
        self.kv = "auto"
        self.attention = "float"
        self.profile = False
        self.threads = 0
        self.save = ""
        self.weights = ""
        var args = argv()
        var a = 1
        while a < len(args):
            var arg = String(args[a])
            if arg == "-v":
                self.verbose = True
            elif arg == "--compare":
                self.compare = True
            elif arg == "--profile":
                self.profile = True
            elif arg.startswith("-") and a + 1 < len(args):
                var val = String(args[a + 1])
                if arg == "--dtype":
                    self.dtype = val
                elif arg == "--ppl":
                    self.ppl = val
                elif arg == "--gguf":
                    self.gguf = val
                elif arg == "--head":
                    self.head = val
                elif arg == "--kv":
                    self.kv = val
                elif arg == "--attention":
                    self.attention = val
                elif arg == "--threads":
                    self.threads = atol(val)
                elif arg == "--save":
                    self.save = val
                elif arg == "--weights":
                    self.weights = val
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


def generate[
    W: WeightMatrix, E: WeightMatrix, KV: KVCache
](args: Args) raises:
    """Loads the model with weights in format W and generates text."""
    var t_load = perf_counter_ns()
    var model = load[W, E, KV](args)
    if args.threads > 0:
        # A barrier needs every team member running at once, so no more
        # than the runtime's worker threads.
        model.set_threads(min(args.threads, parallelism_level()))
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
        model_name[W, E, KV](),
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
    if args.profile:
        model.print_profile()


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
    M: LanguageModel, B: LanguageModel, //, COMPARE: Bool
](model: M, base: B, ids: List[Int], name: String, fmt: String) raises:
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
    print(
        "  ",
        fmt,
        "perplexity",
        ppl,
        " mean NLL",
        totals[NLL] / cnt,
        "nats",
    )
    comptime if COMPARE:
        var ppl_b = exp(totals[NLL_BASE] / cnt)
        print("   f32 perplexity", ppl_b, " change", (ppl / ppl_b - 1) * 100, "%")
        print("   top-1 agreement with f32:", totals[AGREE] / cnt * 100, "%")
        print("   mean KL(f32 || ", fmt, "):", totals[KL] / cnt, "nats")
        print(
            "   logit |diff| vs f32: mean", totals[DIFF_SUM] / (cnt * V),
            " max", diff_max,
        )
        base_logits.unsafe_free()
    logits.unsafe_free()
    stats.unsafe_free()


def evaluate[
    W: WeightMatrix, E: WeightMatrix, KV: KVCache
](args: Args) raises:
    """--ppl: perplexity of the text in args.ppl, and with --compare, how far
    format W's predictions are from float32's."""
    var tok = Tokenizer(args.dir)
    var text = String(from_utf8_lossy=Span(read_file_bytes(args.ppl)))
    var ids = tok.encode(text)
    if len(ids) < 2:
        raise Error("need at least 2 tokens in " + args.ppl)
    var model = load[W, E, KV](args)
    if args.threads > 0:
        # A barrier needs every team member running at once, so no more
        # than the runtime's worker threads.
        model.set_threads(min(args.threads, parallelism_level()))
    var fmt = model_name[W, E, KV]()
    if args.compare:
        # The reference: float32 weights and a float32 KV cache.
        var base = load_model[F32](args.dir + "/model.safetensors")
        eval_loop[True](model, base, ids, args.ppl, fmt)
    else:
        # Without COMPARE the baseline is never used; pass model itself.
        eval_loop[False](model, model, ids, args.ppl, fmt)


def load[
    W: WeightMatrix, E: WeightMatrix, KV: KVCache
](args: Args) raises -> Model[W, E, KV]:
    """Loads the model for formats W, E and KV: from --gguf for GGUF, from
    --weights for a saved model, otherwise from the Hugging Face
    safetensors, converted. With --save, also writes the converted weights."""
    comptime if W.FROM_GGUF:
        if args.save.byte_length() > 0:
            raise Error("--save: GGUF files are already stored converted")
        # W and E are GGUFMatrix here; rebind_var tells the compiler so.
        return rebind_var[Model[W, E, KV]](load_gguf[KV](args.gguf))
    else:
        var model = load_saved[W, E, KV](args.weights) if args.weights.byte_length() > 0 else load_model[W, E, KV](args.dir + "/model.safetensors")
        if args.save.byte_length() > 0:
            save_model(model, args.save)
        return model^


def model_name[W: WeightMatrix, E: WeightMatrix, KV: KVCache]() -> String:
    """The format's name, plus the head's when it differs, plus the KV
    cache's."""
    var n = W.name()
    if W.name() != E.name():
        n += "+head-" + E.name()
    return n + "+kv-" + KV.name()


comptime HEAD8 = QuantMatrix[8, 0, False]
"""The output head format for quantized layers: int8, one scale and zero
point per vocabulary row."""
comptime HEAD8_A16 = QuantMatrix[8, 0, False, True]
"""The same, computed in integers with int16 activations (for -a16 formats)."""


def default_kv(weights: String) -> String:
    """The KV cache format that matches a weight format's precision: float
    formats keep their own type; quantized formats (ours and GGUF) use int8,
    which costs ~0.001 KL and speeds up long contexts (PLAN.md, M5)."""
    if weights == "f32" or weights == "f16" or weights == "bf16":
        return weights
    return "int8"


def run[W: WeightMatrix, E: WeightMatrix = W](args: Args) raises:
    """Picks the KV cache format (--kv, or the default for W) and the
    attention arithmetic (--attention), and runs."""
    var kv = default_kv(W.name()) if args.kv == "auto" else args.kv
    if args.attention == "int":
        if kv != "int8":
            raise Error(
                "--attention int needs the int8 cache (--kv int8 or auto with"
                " quantized weights), not " + kv
            )
        run_with[W, E, IntAttnKV](args)
        return
    if args.attention != "float":
        raise Error("unknown --attention " + args.attention + " (float or int)")
    if kv == "f32":
        run_with[W, E, DenseKV[DType.float32]](args)
    elif kv == "f16":
        run_with[W, E, DenseKV[DType.float16]](args)
    elif kv == "bf16":
        run_with[W, E, DenseKV[DType.bfloat16]](args)
    elif kv == "int16":
        run_with[W, E, QuantKV[16]](args)
    elif kv == "int8":
        run_with[W, E, QuantKV[8]](args)
    else:
        raise Error("unknown --kv " + kv + " (auto, f32, f16, bf16, int16, int8)")


def run_with[W: WeightMatrix, E: WeightMatrix, KV: KVCache](args: Args) raises:
    if args.ppl.byte_length() > 0:
        evaluate[W, E, KV](args)
    else:
        generate[W, E, KV](args)


def run_quantized[W: WeightMatrix](args: Args) raises:
    """Runs a quantized layer format with the head chosen by --head."""
    if args.head == "int8":
        comptime if W.ACT16:
            run[W, HEAD8_A16](args)
        else:
            run[W, HEAD8](args)
    elif args.head == "same":
        run[W, W](args)
    else:
        raise Error("unknown --head " + args.head + " (int8 or same)")


def main() raises:
    var args = Args()
    if args.weights.byte_length() > 0:
        # A saved model's formats come from its file.
        var f = saved_formats(args.weights)
        args.dtype = f[0]
        args.head = "same" if f[1] == f[0] else "int8"
    if args.gguf.byte_length() > 0:
        run[GGUFMatrix](args)
        return
    # Each branch instantiates the whole program for one format, at compile
    # time. Adding a format is one more line here.
    if args.dtype == "f32":
        run[DenseMatrix[DType.float32]](args)
    elif args.dtype == "f16":
        run[DenseMatrix[DType.float16]](args)
    elif args.dtype == "bf16":
        run[DenseMatrix[DType.bfloat16]](args)
    elif args.dtype == "int8-ch":
        run_quantized[QuantMatrix[8, 0, False]](args)
    elif args.dtype == "int8-ch-sym":
        run_quantized[QuantMatrix[8, 0, True]](args)
    elif args.dtype == "int4-ch":
        run_quantized[QuantMatrix[4, 0, False]](args)
    elif args.dtype == "int4-ch-sym":
        run_quantized[QuantMatrix[4, 0, True]](args)
    elif args.dtype == "int4-g128":
        run_quantized[QuantMatrix[4, 128, False]](args)
    elif args.dtype == "int4-g128-sym":
        run_quantized[QuantMatrix[4, 128, True]](args)
    elif args.dtype == "int4-g64":
        run_quantized[QuantMatrix[4, 64, False]](args)
    elif args.dtype == "int4-g64-sym":
        run_quantized[QuantMatrix[4, 64, True]](args)
    elif args.dtype == "int4-g32":
        run_quantized[QuantMatrix[4, 32, False]](args)
    elif args.dtype == "int4-g32-sym":
        run_quantized[QuantMatrix[4, 32, True]](args)
    # W4A16 / W8A16: the same weights, with int16 activations and integer
    # arithmetic (kernels.matmul_rows_a16).
    elif args.dtype == "int8-ch-a16":
        run_quantized[QuantMatrix[8, 0, False, True]](args)
    elif args.dtype == "int4-ch-a16":
        run_quantized[QuantMatrix[4, 0, False, True]](args)
    elif args.dtype == "int4-g128-a16":
        run_quantized[QuantMatrix[4, 128, False, True]](args)
    elif args.dtype == "int4-g64-a16":
        run_quantized[QuantMatrix[4, 64, False, True]](args)
    elif args.dtype == "int4-g32-a16":
        run_quantized[QuantMatrix[4, 32, False, True]](args)
    elif args.dtype == "int4-g32-sym-a16":
        run_quantized[QuantMatrix[4, 32, True, True]](args)
    else:
        raise Error("unknown --dtype " + args.dtype + " (see --help in PLAN.md)")
