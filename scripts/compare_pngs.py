"""Check that regenerated PNG figures match the committed ones pixel for pixel.

    python scripts/compare_pngs.py paper/figures

PNG bytes depend on the zlib build and on the CPU it runs on (Pillow's zlib-ng picks its
compression routines at run time), so the same image can be stored differently on another
machine. CI therefore compares decoded pixels for PNGs and compares the PDF figures byte
for byte.
"""

import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def same_pixels(a: bytes, b: bytes) -> bool:
    x, y = (np.asarray(Image.open(BytesIO(data))) for data in (a, b))
    return x.shape == y.shape and bool(np.array_equal(x, y))


def main(directory: str) -> int:
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    root = Path(top.stdout.strip())
    paths = sorted(Path(directory).glob("*.png"))
    differ = []
    for path in paths:
        name = path.resolve().relative_to(root).as_posix()
        committed = subprocess.run(["git", "show", f"HEAD:{name}"], capture_output=True, cwd=root)
        if committed.returncode != 0 or not same_pixels(committed.stdout, path.read_bytes()):
            differ.append(name)
    for name in differ:
        print(f"DIFFERS  {name}")
    print(f"{len(paths) - len(differ)} of {len(paths)} PNG figures match the committed pixels")
    return 1 if differ or not paths else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
