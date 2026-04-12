"""Project package root for BimanualShift.

This package also exposes vendored third-party dependencies that live under
``third_party/`` so entrypoints can run without requiring separate editable
installs of YARR, RLBench, or PyRep.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_ROOTS = [
    ROOT / "third_party" / "YARR",
    ROOT / "third_party" / "RLBench",
    ROOT / "third_party" / "PyRep",
    ROOT / "third_party" / "pytorch3d",
]

for third_party_root in THIRD_PARTY_ROOTS:
    path_str = str(third_party_root)
    if third_party_root.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
