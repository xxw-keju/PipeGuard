from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from suite.frontend_pairs.sources.triton_cases import write_triton_case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant", required=True, type=int)
    parser.add_argument("--polarity", required=True, choices=("BUG", "CORRECT"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_triton_case(args.output, args.family, args.variant, args.polarity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
