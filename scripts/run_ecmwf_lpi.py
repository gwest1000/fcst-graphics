#!/usr/bin/env python3
"""Archive and produce experimental ECMWF LPI without changing published plots."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import numpy as np
import ecmwf_lpi as ec
import lpi_model


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stamp',required=True,help='Issued run, e.g. 20261007T12Z')
    ap.add_argument('--data-root',type=Path,default=Path('/Volumes/Greg1_2tb/concrete_fcst_data'))
    ap.add_argument('--hours',default='24',help='Comma-separated native forecast end hours; 24,48,...,360 for issued daily windows')
    ap.add_argument('--duration',choices=['3h','24h'],default='24h')
    ap.add_argument('--apply-experimental-transfer',action='store_true',help='Apply the fitted correction, which failed the seven-day held-out assessment')
    ap.add_argument('--archive-only',action='store_true')
    ap.add_argument('--output-dir',type=Path,default=Path('output/ecmwf_lpi'))
    args=ap.parse_args()
    hours=tuple(sorted(set(int(s) for s in args.hours.split(','))))
    if any(h==0 for h in hours) and not args.archive_only:
        ap.error('F000 is an instantaneous diagnostic, excluded from occurrence-window products. Use --archive-only to retain it.')
    if args.duration=='3h' and any(h>144 for h in hours) and not args.archive_only:
        ap.error('No calibrated three-hour probability is available from six-hour snapshots after F144. Use --duration 24h or --archive-only.')
    if args.duration=='24h' and any(h<24 for h in hours) and not args.archive_only:
        ap.error('A daily product requires at least 24 forecast hours.')
    path=ec.ensure_archive(args.data_root,args.stamp,hours)
    print('Regional ingredient archive:',path,flush=True)
    if args.archive_only:
        return
    for hour in hours:
        p,lat,lon,count=ec.forecast(args.data_root,args.stamp,hour,args.duration,args.apply_experimental_transfer)
        output=args.output_dir/args.stamp/f'ecmwf_lpi_{args.duration}_f{hour:03d}.npz'
        ec.atomic_npz(output,potential=p,lat=lat,lon=lon,valid_native_snapshots=count,run_stamp=np.asarray(args.stamp),end_fhour=np.asarray(hour),window_start_fhour=np.asarray(hour-(24 if args.duration=='24h' else 3)),duration=np.asarray(args.duration),experimental=np.asarray(True),probability_scale=np.asarray('percent_of_hrdps_teacher_scale'),teacher_target_radius_km=np.asarray(30),feature_smoothing_sigma_km=np.asarray(20),method=np.asarray('fitted_teacher_transfer' if args.apply_experimental_transfer else 'unadjusted_hrdps_recipe'),teacher_sha256=np.asarray(hashlib.sha256(lpi_model.MODEL_PATH.read_bytes()).hexdigest()),transfer_sha256=np.asarray(hashlib.sha256(ec.MODEL_PATH.read_bytes()).hexdigest()) if args.apply_experimental_transfer else np.asarray(''),interpretation=np.asarray('HRDPS teacher-scale experimental LPI; ECMWF lightning probability has not been independently verified'))
        print(output,flush=True)


if __name__=='__main__':
    main()
