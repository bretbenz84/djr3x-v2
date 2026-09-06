#!/usr/bin/env python3
"""Fetch the pinned CAM++ model without starting Rex or opening audio devices."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if __name__ == '__main__':
    import config
    from audio.campplus import download
    print(download(ROOT / config.CAMPPLUS_MODEL_PATH))
