#!/usr/bin/env python3
"""Install a complete staged HRDPS four-panel LPI run atomically per frame."""
import argparse, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_paths import plot_path
import lpi_model

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stamp',required=True);ap.add_argument('--source-dir',required=True,type=Path);a=ap.parse_args()
 source=a.source_dir/a.stamp;dest=plot_path('hrdps_continental_fourpanel')/a.stamp
 images=[source/f'hrdps_continental_fourpanel_{a.stamp}_f{n:03d}.png' for n in range(0,49,3)]
 for png in images:
  data=png.read_bytes();assert len(data)>10000 and data.endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82'),png
  png.with_suffix('.lpi.json').write_text(json.dumps(dict(formula_version=lpi_model.VERSION,png_sha256=hashlib.sha256(data).hexdigest())))
 dest.mkdir(parents=True,exist_ok=True)
 for png in images:
  for src in (png,png.with_suffix('.lpi.json')):
   target=dest/src.name;fd,tmp=tempfile.mkstemp(prefix='.'+target.name+'.',dir=dest);os.close(fd)
   try:shutil.copy2(src,tmp);os.replace(tmp,target)
   finally:
    if os.path.exists(tmp):os.unlink(tmp)
 print(f'Installed {len(images)} LPI four-panel frames in {dest}')
if __name__=='__main__':main()
