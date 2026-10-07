#!/usr/bin/env python3
"""Regenerate learned LPI verification from available frozen hourly ingredients."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import lpi_model as l
import lightning_ml_archive as archive
import make_hrdps_west_convective as h
import make_hrdps_west_lightning as graphics
import automate_lpi_verification as v

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stamps',required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();cache=args.output_dir/'cache';plots=args.output_dir/'plots';h.set_model('continental');graphics.set_model('continental')
 with np.load(archive.DEFAULT_ARCHIVE_ROOT/'baseline/hrdps_continental_lpi_5km/schema_v1/static/grid.npz') as z:lat=z['lat'];lon=z['lon']
 specs={s.key:s for s in archive.HOURLY_LPI_INGREDIENT_SPECS}
 for stamp in args.stamps.split(','):
  run=h.RunInfo(cycle=stamp[-3:-1],stamp=stamp,init_time=h.parse_stamp(stamp))
  for window in v.full_12z_windows(run):
   arrays=[];files=[archive.hourly_lpi_hour_archive_path(archive.DEFAULT_ARCHIVE_ROOT,stamp,n) for n in range(window.end_fhour-23,window.end_fhour+1)]
   if not all(p.exists() for p in files):print('Unavailable historical hourly window',stamp,window.end_fhour,flush=True);continue
   for p in files:
    with np.load(p) as z:arrays.append(np.stack([archive.unpack_field(z[k],specs[k]) for k in l.KEYS]))
   a=np.stack(arrays);paths={}
   for j,end in enumerate(window.included_hours):
    p,_=l.infer(a[j*3:j*3+3],'3h');paths[end]=graphics.save_lpi_cache(cache/stamp/f'hrdps_continental_lightning_{stamp}_f{end:03d}.png',run,end,lat,lon,p,1)
   p,count=l.infer(a,'24h');daily=Path(str(paths[window.end_fhour]).replace('_lpi.npz','_lpi24h.npz'));l.write_cache(daily,run,'24h',lat,lon,p,count,end_hour=window.end_fhour)
 config=v.parse_args(['--continental-cache-dir',str(cache),'--continental-output-dir',str(plots),'--stamps',args.stamps,'--force','--no-publish'])
 results=v.render_ready_verifications(config);assert results,'No observation-complete verification could be regenerated';print('VERIFIED',*[str(p) for p in results],flush=True)
if __name__=='__main__':main()
