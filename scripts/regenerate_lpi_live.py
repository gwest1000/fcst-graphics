#!/usr/bin/env python3
"""Regenerate only LPI/fire-weather products, preserving other forecast products."""
import os,sys,argparse,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')

def render(task):
 stamp,hour,data_dir,out=task
 import make_hrdps_west_convective as h
 import make_hrdps_west_lightning as l
 h.set_model('continental');l.set_model('continental');run=h.RunInfo(cycle=stamp[-3:-1],stamp=stamp,init_time=h.parse_stamp(stamp))
 return [str(p) for p in l.make_region_plots(run,Path(data_dir),Path(out),2,5,7,hours=(hour,),region_keys=('bc',),render_wind=False)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stamp',required=True);ap.add_argument('--data-dir',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
 import make_hrdps_west_convective as h
 import make_hrdps_west_lightning as l
 import lpi_model
 h.set_model('continental');l.set_model('continental')
 tasks=[(args.stamp,n,args.data_dir,args.output_dir) for n in range(0,49,3)]
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
  for task,result in zip(tasks,ex.map(render,tasks)):print('DONE',task[1],result,flush=True)
 # Assemble daily windows only after every parallel worker has retained its inputs.
 import numpy as np
 run=h.RunInfo(cycle=args.stamp[-3:-1],stamp=args.stamp,init_time=h.parse_stamp(args.stamp));folder=Path(args.output_dir)/args.stamp
 sample=l.hour_file(Path(args.data_dir)/args.stamp,run,3,'MU-VT-LI','ISBL','500');_,lat,lon=h.read_grib(sample,coords=True);ys,xs=h.subset_slices(lat,lon,h.model_config().extent);la,lo=lat[ys,xs],lon[ys,xs]
 for end in lpi_model.daily_end_hours(run.cycle):
  p=folder/'lpi_cache'/f'hrdps_continental_lightning_{args.stamp}_f{end:03d}_lpi.npz';direct=l.make_learned_daily_cache(p,run,None,la,lo,end_hour=end,input_dir=folder/'lpi_cache/ingredients');assert direct is not None
  import datetime as dt
  from dataclasses import replace
  with np.load(direct) as z:lpi_model.plot_daily(folder/f'hrdps_continental_lightning_daily_{args.stamp}_f{end:03d}.png',replace(run,init_time=run.init_time+dt.timedelta(hours=end-24)),z['lat'],z['lon'],z['potential'])
 print('ALL48 COMPLETE',flush=True)
if __name__=='__main__':main()
