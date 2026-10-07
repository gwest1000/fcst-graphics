#!/usr/bin/env python3
"""Generate a direct day-one LPI forecast from retained hourly ingredients."""
import argparse,sys,json,datetime as dt
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import lpi_model as l
import lightning_ml_archive as archive
import make_hrdps_west_convective as hrdps

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stamp',required=True);ap.add_argument('--archive-root',type=Path,default=archive.DEFAULT_ARCHIVE_ROOT);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
 init=hrdps.parse_stamp(args.stamp)
 if init.hour!=12:raise ValueError('The deployed model supports 12Z continental HRDPS only')
 run=hrdps.RunInfo(cycle='12',stamp=args.stamp,init_time=init);specs={s.key:s for s in archive.HOURLY_LPI_INGREDIENT_SPECS};arrays=[]
 for hour in range(1,25):
  path=archive.hourly_lpi_hour_archive_path(args.archive_root,args.stamp,hour)
  with np.load(path) as z:
   if str(z['grid_hash'].item())!=l.load_model()['grid_hash']:raise ValueError('Hourly input grid mismatch')
   arrays.append(np.stack([archive.unpack_field(z[k],specs[k]) for k in l.KEYS]))
 grid=args.archive_root/'baseline/hrdps_continental_lpi_5km/schema_v1/static/grid.npz'
 with np.load(grid) as z:lat=z['lat'];lon=z['lon']
 if l.grid_hash(lat,lon)!=l.load_model()['grid_hash']:raise ValueError('Coordinate grid mismatch')
 args.output_dir.mkdir(parents=True,exist_ok=True);p,count=l.infer(np.stack(arrays),'24h');path=args.output_dir/f'lpi_{args.stamp}_f024_lpi24h.npz';l.write_cache(path,run,'24h',lat,lon,p,count)
 png=args.output_dir/f'lpi_{args.stamp}_24h.png';l.plot_daily(png,run,lat,lon,p)
 summary=dict(stamp=args.stamp,version=l.VERSION,cache=str(path),map=str(png),valid_cells=int(np.isfinite(p).sum()),peak_probability=float(np.nanmax(p)) if np.isfinite(p).any() else None,source_hours=24)
 (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
