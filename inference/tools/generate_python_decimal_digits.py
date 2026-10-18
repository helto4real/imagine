#!/usr/bin/env python3
"""Source Python Unicode decimal-digit table, independent of tensor/runtime data."""
import argparse,subprocess,sys,unicodedata
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();assert sys.version_info[:3]==(3,12,13);assert unicodedata.unidata_version=='15.0.0'
zeros=[n for n in range(0x110000)if unicodedata.decimal(chr(n),None)==0]
for n in zeros:assert[unicodedata.decimal(chr(n+i))for i in range(10)]==list(range(10))
s='// Generated from the frozen Python 3.12.13 Unicode 15.0.0 decimal table.\npub const ZEROS:&[u32]=&['+','.join(map(str,zeros))+'];\n';s=subprocess.run(['rustfmt','--edition','2024','--emit','stdout'],input=s,text=True,capture_output=True,check=True).stdout
if a.check:assert a.output.read_text()==s
else:a.output.write_text(s)
