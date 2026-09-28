"""GPT-2 byte-level BPE tokenizer.

Reads the Hugging Face vocab.json and merges.txt. `encode` splits text with
GPT-2's pre-tokenizer regex (implemented by hand in `pretokenize`), then
applies BPE merges within each chunk. `token_bytes[id]` gives the raw bytes of
a token for decoding.
"""

from std.collections import Dict

comptime VOCAB_SIZE = 50257


def read_file_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    return b^


def parse_uint(b: Span[UInt8, _], mut i: Int) -> Int:
    """Parses decimal digits starting at b[i], advancing i past them."""
    var n = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        n = n * 10 + Int(b[i] - 48)
        i += 1
    return n


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
        self.token_bytes = List[List[UInt8]](length=VOCAB_SIZE, fill=List[UInt8]())
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
