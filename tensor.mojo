"""Weight storage formats.

A weight matrix is always *used* as float32, but it can be *stored* however a
format likes: float32, float16, quantized integers, ... Each format is a
struct implementing the `WeightMatrix` trait, and the kernels in kernels.mojo
are generic over that trait.

Mojo has no class inheritance. A trait is an interface, and a function or
struct generic over it (`def f[W: WeightMatrix](w: W)`) is compiled separately
for each concrete format, with the format's methods inlined. So there is no
virtual-call cost in the inner loops: `load` on a float32 matrix compiles to
exactly the plain SIMD load the untyped gpt2.mojo does.
"""

from std.memory.alloc import unsafe_alloc
from std.sys import simd_width_of, size_of

comptime FPtr = Pointer[Float32, MutUntrackedOrigin]
comptime NW = simd_width_of[DType.float32]()
comptime F32V = SIMD[DType.float32, NW]


trait WeightMatrix(Deinitable, ImplicitlyCopyable):
    """A [rows, cols] matrix of weights, read back as float32.

    Values are handles: copying one copies the pointer, not the data, and the
    owner releases the storage by calling `free` once. That keeps them cheap to
    pass to kernels and to store in a List.
    """

    comptime NAME: StaticString
    """Short name for the format, as used by --dtype."""

    @staticmethod
    def from_f32(src: FPtr, rows: Int, cols: Int) -> Self:
        """Converts a row-major float32 matrix into this format (a copy)."""
        ...

    def load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]:
        """Returns `width` consecutive values of `row`, starting at `col`.

        `col` must be a multiple of `width`.
        """
        ...

    def nbytes(self) -> Int:
        """Bytes of storage, to report the model's size."""
        ...

    def free(self):
        """Releases the storage."""
        ...


struct DenseMatrix[dtype: DType](WeightMatrix):
    """Every element stored as `dtype` (float32, float16, bfloat16, ...).

    `load` widens to float32 with `.cast`, which is free for float32, one
    VCVTPH2PS for float16, and a shift for bfloat16 (see
    research/test_half.mojo).
    """

    comptime NAME = "f32" if Self.dtype == DType.float32 else (
        "f16" if Self.dtype == DType.float16 else "bf16"
    )

    var data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var rows: Int
    var cols: Int

    def __init__(
        out self,
        data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
        rows: Int,
        cols: Int,
    ):
        self.data = data
        self.rows = rows
        self.cols = cols

    @staticmethod
    def from_f32(src: FPtr, rows: Int, cols: Int) -> Self:
        var n = rows * cols
        var data = unsafe_alloc[Scalar[Self.dtype]](n)
        for i in range(0, n - n % NW, NW):
            data.unsafe_store(
                i, src.unsafe_load[width=NW](i).cast[Self.dtype]()
            )
        for i in range(n - n % NW, n):
            data[unsafe_offset=i] = src[unsafe_offset=i].cast[Self.dtype]()
        return Self(data, rows, cols)

    @always_inline
    def load[width: Int](self, row: Int, col: Int) -> SIMD[DType.float32, width]:
        return self.data.unsafe_load[width=width](row * self.cols + col).cast[
            DType.float32
        ]()

    def nbytes(self) -> Int:
        return self.rows * self.cols * size_of[Scalar[Self.dtype]]()

    def free(self):
        self.data.unsafe_free()
