"""Writing and reading raw model data: integers, strings and byte buffers.

Weight formats save and restore themselves through these (WeightMatrix.save
and restore, tensor.mojo), and gpt2t.mojo builds its saved model files on
them (M7). Both stream straight to and from the file: buffers are written
from and read into their own memory, with no copy in between.

Integers are 8 bytes, little-endian; strings are a length and the bytes.
"""

from std.io.file import FileHandle
from std.memory.alloc import unsafe_alloc
from std.sys import size_of

comptime BPtr = Pointer[UInt8, MutUntrackedOrigin]


struct ByteWriter(Movable):
    """Writes to a file opened for writing."""

    var f: FileHandle
    var written: Int  # bytes so far

    def __init__(out self, path: String) raises:
        self.f = open(path, "w")
        self.written = 0

    def int(mut self, v: Int) raises:
        var b = SIMD[DType.uint8, 8]()
        comptime for i in range(8):
            b[i] = UInt8((v >> (8 * i)) & 0xFF)
        var tmp = unsafe_alloc[UInt8](8)
        tmp.unsafe_store(0, b)
        self.buffer(tmp, 8)
        tmp.unsafe_free()

    def string(mut self, s: String) raises:
        var n = s.byte_length()
        self.int(n)
        var tmp = unsafe_alloc[UInt8](max(n, 1))
        var src = s.as_bytes()
        for i in range(n):
            tmp[unsafe_offset=i] = src[i]
        self.buffer(tmp, n)
        tmp.unsafe_free()

    def buffer[
        dtype: DType, //
    ](mut self, p: Pointer[Scalar[dtype], MutUntrackedOrigin], count: Int) raises:
        """Writes count elements starting at p."""
        var n = count * size_of[Scalar[dtype]]()
        if n > 0:
            self.f.write_bytes(Span(unsafe_ptr=p.unsafe_bitcast[UInt8](), length=n))
        self.written += n

    def close(mut self) raises:
        self.f.close()


struct ByteReader(Movable):
    """Reads what a ByteWriter wrote, in the same order."""

    var f: FileHandle
    var path: String

    def __init__(out self, path: String) raises:
        self.f = open(path, "r")
        self.path = path

    def fill(mut self, p: BPtr, n: Int) raises:
        """Reads exactly n bytes into p."""
        var done = 0
        while done < n:
            var got = self.f.read(
                Span(unsafe_ptr=p.unsafe_offset(done), length=n - done)
            )
            if got <= 0:
                raise Error("unexpected end of " + self.path)
            done += got

    def int(mut self) raises -> Int:
        var tmp = unsafe_alloc[UInt8](8)
        self.fill(tmp, 8)
        var v = 0
        for i in range(8):
            v |= Int(tmp[unsafe_offset=i]) << (8 * i)
        tmp.unsafe_free()
        return v

    def string(mut self) raises -> String:
        var n = self.int()
        if n < 0 or n > 1 << 20:
            raise Error("bad string length in " + self.path)
        return self.chars(n)

    def chars(mut self, n: Int) raises -> String:
        """Reads n bytes as a string."""
        var tmp = unsafe_alloc[UInt8](max(n, 1))
        self.fill(tmp, n)
        var s = String(from_utf8_lossy=Span(unsafe_ptr=tmp, length=n))
        tmp.unsafe_free()
        return s

    def buffer[
        dtype: DType
    ](mut self, count: Int) raises -> Pointer[Scalar[dtype], MutUntrackedOrigin]:
        """Allocates count elements and reads them in; the caller owns (frees)
        the result."""
        var p = unsafe_alloc[Scalar[dtype]](max(count, 1))
        self.fill(p.unsafe_bitcast[UInt8](), count * size_of[Scalar[dtype]]())
        return p

    def close(mut self) raises:
        self.f.close()
