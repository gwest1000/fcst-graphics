#!/usr/bin/env python3
"""Regenerate the current HRDPS four-panel run with the shared learned LPI."""
import argparse, concurrent.futures, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def render(task):
 stamp,hour,data,out=task
 import make_hrdps_west_fourpanel as f
 import make_hrdps_west_convective as h
 f.set_model('continental')
 run=h.RunInfo(cycle=stamp[-3:-1],stamp=stamp,init_time=h.parse_stamp(stamp))
 return [str(p) for p in f.make_plots(run,Path(data),Path(out),h.WATERSHED_CACHE,False,False,2,5,7,hours=(hour,))]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stamp',required=True);ap.add_argument('--data-dir',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
 tasks=[(a.stamp,n,a.data_dir,a.output_dir) for n in range(0,49,3)]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  for task,result in zip(tasks,ex.map(render,tasks)):print('DONE',task[1],result,flush=True)
if __name__=='__main__':main()
