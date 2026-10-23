#!/usr/bin/env python3
"""Compatibility entry point; prints the verified source path (not a fixed path)."""
import subprocess
import sys
from pathlib import Path
subprocess.run([sys.executable, str(Path(__file__).with_name('product_build.py')),
                'native-source'], check=True)
