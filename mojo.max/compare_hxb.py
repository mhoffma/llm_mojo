"""Compare two .hxb containers section by section (header and section table: hexagon_torch/export_blob.py).

  python compare_hxb.py a.hxb b.hxb

For each tag: length, CRC32 and whether the bytes are identical; for differing sections, how many bytes
differ. The metadata section (JSON) is also compared key by key.
"""
import json
import struct
import sys

import numpy as np

NAMES = {1: "metadata", 2: "weights arena", 3: "head weights", 4: "embedding", 5: "decode schedule", 6: "head schedule",
         7: "prefill schedules", 11: "prefill placement", 12: "extra arenas", 13: "prefill patch", 14: "kv init"}


def sections(path):
    with open(path, "rb") as f:
        magic, ver, hsz, n, _fl, total, first, _ = struct.unpack("<4sIIIIQQ28s", f.read(64))
        assert magic == b"HXMB"
        table = [struct.unpack("<IIQQII", f.read(32)) for _ in range(n)]
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return {t[0]: (t[2], t[3], t[4], mm) for t in table}


def main(pa, pb):
    a, b = sections(pa), sections(pb)
    ok = True
    for tag in sorted(set(a) | set(b)):
        name = NAMES.get(tag, f"tag {tag}")
        if tag not in a or tag not in b:
            print(f"{name:20s} only in {'A' if tag in a else 'B'}")
            ok = False
            continue
        (oa, la, ca, ma), (ob, lb, cb, mb) = a[tag], b[tag]
        if la != lb:
            print(f"{name:20s} length {la} vs {lb}")
            ok = False
            continue
        xa, xb = np.asarray(ma[oa:oa + la]), np.asarray(mb[ob:ob + lb])
        nd = int(np.count_nonzero(xa != xb))
        print(f"{name:20s} {la:>12d} bytes  crc {'same' if ca == cb else 'DIFFERENT'}  " + ("identical" if nd == 0 else f"{nd} bytes differ"))
        ok &= nd == 0
        if tag == 1 and nd:
            ja, jb = json.loads(bytes(xa)), json.loads(bytes(xb))
            for k in sorted(set(ja) | set(jb)):
                if ja.get(k) != jb.get(k):
                    print(f"    metadata[{k!r}]: {str(ja.get(k))[:80]} vs {str(jb.get(k))[:80]}")
    print("containers identical" if ok else "containers differ")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
