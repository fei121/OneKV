#!/usr/bin/env python3
"""CLI entry point for the 3-state token trace generator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from onekv.trace import main
if __name__ == "__main__":
    main()
