#!/usr/bin/env python3
"""Atomically install staged, validated LPI products without removing other files."""
import argparse, os, shutil, sys, tempfile, json, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from project_paths import plot_path

def check_cache(path,stamp,hour,daily=False):
 with np.load(path) as z:
  assert str(z['run_stamp'].item())==stamp, path
  assert int(z['fhour'].item())==hour, path
  assert str(z['formula_version'].item()).startswith('bc_lpi_v4_random32'), path
  if daily: assert np.array_equal(z['window_fhours'],np.arange(hour-23,hour+1)), path

def install_tree(source,target):
 count=0
 for src in sorted(source.rglob('*')):
  if not src.is_file(): continue
  dst=target/src.relative_to(source);dst.parent.mkdir(parents=True,exist_ok=True)
  fd,tmp=tempfile.mkstemp(prefix='.'+dst.name+'.',dir=dst.parent);os.close(fd)
  try: shutil.copy2(src,tmp);os.replace(tmp,dst)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
  count+=1
 print(f'Installed {count} files in {target}',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--forecast-dir',type=Path,required=True);ap.add_argument('--stamp',required=True);ap.add_argument('--verification-dir',type=Path,required=True);a=ap.parse_args()
 root=a.forecast_dir/a.stamp
 for hour in range(0,49,3):
  assert (root/f'hrdps_continental_lightning_twopanel_{a.stamp}_f{hour:03d}.png').stat().st_size>10000
  check_cache(root/'lpi_cache'/f'hrdps_continental_lightning_{a.stamp}_f{hour:03d}_lpi.npz',a.stamp,hour)
 for hour in range(0,49,3):
  cache=root/'lpi_cache'/f'hrdps_continental_lightning_{a.stamp}_f{hour:03d}_lpi.npz'
  png=root/f'hrdps_continental_lightning_twopanel_{a.stamp}_f{hour:03d}.png'
  with np.load(cache) as z: version=str(z['formula_version'].item())
  cache.with_suffix('.display.json').write_text(json.dumps(dict(formula_version=version,png_sha256=hashlib.sha256(png.read_bytes()).hexdigest())))
 for hour in (24,48): check_cache(root/'lpi_cache'/f'hrdps_continental_lightning_{a.stamp}_f{hour:03d}_lpi24h.npz',a.stamp,hour,True)
 # Validate every verification image's matching daily forecast before changing destinations.
 verification=[]
 for src in sorted((a.verification_dir/'plots').glob('*')):
  if not src.is_dir():continue
  stamp=src.name
  for image in src.glob('*.png'):
   hour=int(image.stem.rsplit('_f',1)[1]);assert image.stat().st_size>10000
   check_cache(a.verification_dir/'cache'/stamp/'lpi_cache'/f'hrdps_continental_lightning_{stamp}_f{hour:03d}_lpi24h.npz',stamp,hour,True)
  verification.append((stamp,src))
 assert verification
 install_tree(root,plot_path('hrdps_continental_lightning')/a.stamp)
 for stamp,src in verification:
  install_tree(a.verification_dir/'cache'/stamp,plot_path('hrdps_continental_lightning')/stamp)
  install_tree(src,plot_path('hrdps_continental_lightning_verif')/stamp)
if __name__=='__main__':main()
