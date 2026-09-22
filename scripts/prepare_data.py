"""Source-checkout entry point for checkpoint-free HOIGPT data preparation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoigpt.preprocessing.cli import main


if __name__ == '__main__':
    main()
