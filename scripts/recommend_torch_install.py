from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_utils import collect_runtime_info, dumps_pretty


def main() -> None:
    info = collect_runtime_info()
    print(dumps_pretty(info))
    print()
    print("Recommended install command:")
    print(info["recommended_torch_install"])


if __name__ == "__main__":
    main()
