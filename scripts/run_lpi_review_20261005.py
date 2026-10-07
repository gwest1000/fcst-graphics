#!/usr/bin/env python3
"""Reproducible, read-only archive study of LPI timing, location and season."""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/fcstgraphics-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union

import analyze_lpi_calibration as old
import lightning_ml_archive as archive
import project_paths

UTC = dt.timezone.utc
SCALES = (0., 5., 10., 15., 20.)
RADII = (10., 20., 30., 40.)
LAGS = (-6, -3, 0, 3, 6)
VERSION = 'bc_lpi_v3_3hmax'
SEED = 20261005
NBINS = 200


def log(s): print(s, flush=True)
def parse(s): return dt.datetime.strptime(s, '%Y%m%dT%HZ').replace(tzinfo=UTC)
def obs_parse(p): return dt.datetime.strptime(p.name[:13], '%Y%m%dT%H%M').replace(tzinfo=UTC)
def day(t): return (t - dt.timedelta(hours=12, microseconds=1)).date().isoformat()
def iso(t): return t.isoformat().replace('+00:00', 'Z')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p, v): p.write_text(json.dumps(v, indent=2, default=str, allow_nan=True)+'\n')
def write_csv(p, rows):
    if not rows: p.write_text(''); return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def smooth(a, km):
    if km == 0: return a.copy()
    good=np.isfinite(a); sig=km/5.
    weights=gaussian_filter(good.astype(np.float32), sig, mode='constant', truncate=4.)
    values=gaussian_filter(np.where(good,a,0).astype(np.float32), sig, mode='constant', truncate=4.)
    return np.divide(values,weights,out=np.full(a.shape,np.nan,np.float32),where=(weights>=.95)&good)


def lead_pair(stamp, h): return ((parse(stamp)-dt.timedelta(days=1)).strftime('%Y%m%dT%HZ'),h+24)
def lag_end(valid, delta): return valid+dt.timedelta(hours=delta)


def observation(p, lon, lat, xy):
    """Distances to flash-cell centers in EPSG:3005; missing coverage stays missing."""
    tr=Transformer.from_crs(4326,3005,always_xy=True)
    with rasterio.open(p) as src:
        if src.crs.to_epsg()!=4326: raise RuntimeError(f'Unexpected observation CRS: {src.crs}')
        win=src.window(float(lon.min()-1.5),float(lat.min()-1.),float(lon.max()+1.5),float(lat.max()+1.)).round_offsets().round_lengths()
        a=src.read(1,window=win,boundless=True,fill_value=src.nodata).astype(np.float32)
        affine=src.window_transform(win)
        valid=np.isfinite(a) & (a!=src.nodata)
        points=np.column_stack(xy)
        def dist(mask):
            rr,cc=np.nonzero(mask)
            if not len(rr): return np.full(len(lon),np.inf)
            llon,llat=rasterio.transform.xy(affine,rr,cc,offset='center')
            xx,yy=tr.transform(llon,llat)
            return cKDTree(np.column_stack((xx,yy))).query(points,workers=1)[0]/1000.
        flash_distance=dist(valid & (a>0))
        invalid_distance=dist(~valid)
        rr,cc=rasterio.transform.rowcol(affine,lon,lat); rr=np.asarray(rr); cc=np.asarray(cc)
        inside=(rr>=0)&(rr<a.shape[0])&(cc>=0)&(cc<a.shape[1])
        coverage=np.zeros(len(lon),bool); density=np.full(len(lon),np.nan,np.float32)
        coverage[inside]=valid[rr[inside],cc[inside]]
        density[coverage]=a[rr[coverage],cc[coverage]]
        targets=np.full((len(RADII),len(lon)),255,np.uint8)
        for k,r in enumerate(RADII):
            safe=coverage & (invalid_distance>r+3.)
            targets[k,safe]=(flash_distance[safe]<=r).astype(np.uint8)
    return targets,density


def prepare(root, out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'prepared.json').exists():
        log('Reusing frozen preparation'); return
    obspaths=sorted((root/'observations/eccc_lightning_3h/schema_v1').glob('*/*/*.tif'))
    latest=max(map(obs_parse,obspaths)); cutoff=latest.replace(hour=12,minute=0)
    if latest<cutoff: cutoff-=dt.timedelta(days=1)
    obspaths=[p for p in obspaths if obs_parse(p)<=cutoff]
    obsrows=[{'index':i,'valid':iso(obs_parse(p)),'day':day(obs_parse(p)),'path':str(p),'sha256':digest(p)} for i,p in enumerate(obspaths)]
    base=root/'baseline/hrdps_continental_lpi_5km/schema_v1'
    with np.load(base/'static/grid.npz') as g:
        lat=g['lat']; lon=g['lon']; grid_hash=str(g['grid_hash'].item())
    la,lo,bc,cor=old.domain_masks(root)
    assert lat.shape==la.shape and np.allclose(lat,la) and np.allclose(lon,lo),'Grid mismatch'
    coast_data=json.loads(Path('data/bc_danger_regions/natural_resource_districts.geojson').read_text())
    coast_geom=unary_union([shape(f['geometry']) for f in coast_data['features'] if f['properties']['REGION_ORG_UNIT'] in ('RSC','RWC') or f['properties']['ORG_UNIT']=='DKM'])
    coastal=contains_xy(coast_geom,lon,lat)
    recs=[]; inventory=[]; common=np.ones(lat.shape,bool); versions=collections.Counter()
    for p in sorted(base.glob('*/*/f*.npz')):
        h=int(p.stem[1:]); init=parse(p.parent.name); valid=init+dt.timedelta(hours=h)
        side=p.with_suffix('.json'); meta=json.loads(side.read_text()) if side.exists() else {}
        version=meta.get('formula_version','unknown'); versions[version]+=1
        reason='instantaneous F000' if h==0 else 'older formula' if version!=VERSION else 'after cutoff' if valid>cutoff else ''
        row={'path':str(p),'run':p.parent.name,'cycle':init.hour,'fhour':h,'init':iso(init),'valid':iso(valid),'day':day(valid),'horizon':1 if h<=24 else 2,'version':version,'grid_hash':meta.get('cache_grid_hash'),'reason':reason}
        if not reason:
            sha=digest(p)
            if meta.get('sha256')!=sha: raise RuntimeError(f'Checksum mismatch or missing sidecar: {p}')
            with np.load(p) as z:
                a=z['potential_x2']; finite=a!=255
                assert str(z['cache_grid_hash'].item())==grid_hash
                assert int(z['forecast_hour'][0])==h and int(z['valid_unix'][0])==int(valid.timestamp())
                assert a.shape==lat.shape
            row['finite_bc_fraction']=float(finite[bc].mean())
            row['sha256']=sha; row['index']=len(recs); recs.append(row.copy())
        inventory.append(row)
    # Explicit zero padding prevents distance-transform edge extrapolation.
    support=distance_transform_edt(np.pad(np.ones(lat.shape,bool),1,constant_values=False),sampling=5.)[1:-1,1:-1]
    mask=bc & (support>80.)
    ids=np.flatnonzero(mask); flat_lat=lat.ravel()[ids]; flat_lon=lon.ravel()[ids]
    if not np.any(cor.ravel()[ids]): raise RuntimeError('No common supported corridor cells')
    np.savez(out/'grid.npz',lat=lat,lon=lon,indices=ids,cor=cor.ravel()[ids],coastal=coastal.ravel()[ids],support=support,mask=mask,original_bc_cells=bc.sum(),original_corridor_cells=cor.sum())
    dates=sorted({r['day'] for r in recs if r['valid'] in {o['valid'] for o in obsrows}})
    t1=int(len(dates)*.6); t2=int(len(dates)*.8)
    bounds=[dt.date.fromisoformat(dates[t1]),dt.date.fromisoformat(dates[t2])]
    def splitdate(d):
        d=dt.date.fromisoformat(d)
        if any(abs((d-b).days)<=1 for b in bounds): return 'purged'
        return 'train' if d<bounds[0] else 'validate' if d<bounds[1] else 'test'
    lookup={d:splitdate(d) for d in dates}
    for r in recs:
        init=dt.datetime.fromisoformat(r['init'].replace('Z','+00:00'))
        run_splits={splitdate(day(init+dt.timedelta(hours=h))) for h in range(3,49,3)}
        split=splitdate(r['day'])
        r['split']=split if run_splits=={split} and split!='purged' else 'purged'
    pre={'cutoff':iso(cutoff),'latest_observation_at_freeze':iso(latest),'grid_hash':grid_hash,'shape':list(lat.shape),'versions':dict(versions),'dates':dates,'splits':lookup,'boundaries':[str(x) for x in bounds],'seed':SEED,'sigmas_km':SCALES,'radii_km':RADII,'lags_hours':LAGS,'mask_bc_cells':len(ids),'mask_corridor_cells':int(cor.ravel()[ids].sum()),'original_bc_cells':int(bc.sum()),'original_corridor_cells':int(cor.sum()),'observation_method':'Nearest positive raster cell center in EPSG:3005; radius requires no nodata center within radius + 3 km','region_proxy':'South Coast and West Coast Natural Resource Regions plus Coast Mountains district; geographic proxy, not meteorological regime','split_purge':'Within one calendar day of each boundary; entire issued runs spanning splits excluded','code_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'script_sha256':digest(Path(__file__))}
    write_json(out/'manifest.json',pre); write_json(out/'records.json',recs); write_json(out/'observations.json',obsrows)
    write_csv(out/'sample_inventory.csv',inventory)
    np.savez(out/'coordinates.npz',lat=flat_lat,lon=flat_lon)
    log(f'Frozen {len(recs)} v3 fields; {len(obsrows)} obs; {len(ids)} supported BC cells; cutoff {cutoff}')
    scores=np.lib.format.open_memmap(out/'scores.npy',mode='w+',dtype=np.float16,shape=(len(recs),len(SCALES),len(ids)))
    for i,r in enumerate(recs):
        with np.load(r['path']) as z: a=z['potential_x2'].astype(np.float32)*.5; a[z['potential_x2']==255]=np.nan
        for k,s in enumerate(SCALES): scores[i,k]=smooth(a,s).ravel()[ids]
        if i%250==0: log(f'Forecast smoothing {i}/{len(recs)}')
    scores.flush()
    targets=np.lib.format.open_memmap(out/'targets.npy',mode='w+',dtype=np.uint8,shape=(len(obsrows),len(RADII),len(ids)))
    density=np.lib.format.open_memmap(out/'density.npy',mode='w+',dtype=np.float16,shape=(len(obsrows),len(ids)))
    xy=Transformer.from_crs(4326,3005,always_xy=True).transform(flat_lon,flat_lat)
    summaries=[]
    for i,r in enumerate(obsrows):
        t,d=observation(Path(r['path']),flat_lon,flat_lat,xy); targets[i]=t; density[i]=d
        m=cor.ravel()[ids]; valid=t[2,m]!=255
        summaries.append({'valid':r['valid'],'day':r['day'],'corridor_valid_cells':int(valid.sum()),'corridor_event_cells_30km':int((t[2,m]==1).sum()),'corridor_event_fraction_30km':float((t[2,m][valid]==1).mean()) if valid.any() else np.nan,'corridor_density_sum':float(np.nansum(d[m]))})
        if i%40==0: log(f'Observation physical-distance targets {i}/{len(obsrows)}')
    targets.flush(); density.flush(); write_csv(out/'observation_inventory.csv',summaries)
    write_json(out/'prepared.json',{'complete':True,'records':len(recs),'observations':len(obsrows)})


class Study:
    def __init__(self,out):
        self.out=out; self.man=json.loads((out/'manifest.json').read_text()); self.records=json.loads((out/'records.json').read_text()); self.obs=json.loads((out/'observations.json').read_text())
        self.scores=np.load(out/'scores.npy',mmap_mode='r'); self.targets=np.load(out/'targets.npy',mmap_mode='r'); self.density=np.load(out/'density.npy',mmap_mode='r')
        with np.load(out/'grid.npz') as z: self.grid={k:z[k] for k in z.files}
        with np.load(out/'coordinates.npz') as z: self.lat=z['lat']; self.lon=z['lon']
        self.cor=self.grid['cor']; self.coast=self.grid['coastal']; self.rec_by={(r['run'],r['fhour']):r for r in self.records}; self.obs_by={r['valid']:r['index'] for r in self.obs}
        self.results=[]; self.bundles={}; self.support=[]; self.selection={}; self.solar_cache={}
    def solar(self,valid):
        if valid in self.solar_cache: return self.solar_cache[valid]
        t=dt.datetime.fromisoformat(valid.replace('Z','+00:00')); mid=t-dt.timedelta(hours=1.5)
        elevation=old.solar_elevation(mid,self.lat,self.lon)
        maximum,daylength=old.solar_max_elevation_and_daylight_hours(mid,self.lat)
        decl=old.solar_declination(mid)
        hour=(mid.hour+mid.minute/60+self.lon/15)%24
        daylight=np.mean([old.solar_elevation(t-dt.timedelta(hours=h),self.lat,self.lon)>0 for h in (.5,1.5,2.5)],axis=0)
        v={'elevation':elevation,'maximum':maximum,'daylength':daylength,'declination':decl,'hour':hour,'daylight':daylight,'sin':np.sin(hour*np.pi/12),'cos':np.cos(hour*np.pi/12),'calendar':(mid-dt.datetime(2026,7,1,tzinfo=UTC)).days/90}
        self.solar_cache[valid]=v; return v
    def basecases(self):
        return [{'record':r,'obs':self.obs_by[r['valid']],'anchor':r['valid'],'day':r['day'],'split':r['split'],'horizon':r['horizon']} for r in self.records if r['split']!='purged' and r['valid'] in self.obs_by]
    def groups(self,c):
        r=c['record']; solar=self.solar(c['anchor']); cor=self.cor
        return {'corridor':cor,'all_bc':np.ones(len(cor),bool),'coastal_proxy':cor&self.coast,'interior_proxy':cor&~self.coast,'night':cor&(solar['elevation']<-6),'twilight':cor&(solar['elevation']>=-6)&(solar['elevation']<0),'daylight':cor&(solar['elevation']>=0),f'cycle_{r["cycle"]:02d}':cor,f'lead_{((r["fhour"]-1)//12)*12+1:02d}_{((r["fhour"]-1)//12+1)*12:02d}':cor,f'month_{c["anchor"][5:7]}':cor}
    def evaluate(self,name,cases,names,getter,stratify=False,probability=False):
        nbin=1000 if probability else NBINS
        mult=collections.Counter((c['anchor'],c['horizon']) for c in cases)
        hist={}; perday={}; records=collections.Counter(); active_dates=collections.defaultdict(set); eligible_dates=collections.defaultdict(set); eligible_blocks=collections.defaultdict(set)
        for j,c in enumerate(cases):
            vals=getter(c)
            if vals is None: continue
            common=np.logical_and.reduce([np.isfinite(s)&np.isfinite(y)&(y!=255) for s,y in vals.values()])
            masks=self.groups(c) if stratify else {'corridor':self.cor}
            weight=1./mult[(c['anchor'],c['horizon'])]
            for group,mask in masks.items():
                use=common&mask
                if not use.any(): continue
                for cand,(score,event) in vals.items():
                    key=(c['horizon'],cand,c['split'],group)
                    h=hist.setdefault(key,np.zeros((2,nbin),float))
                    bins=np.clip((score[use]*(nbin if probability else nbin/100)).astype(int),0,nbin-1)
                    count=np.bincount(bins,minlength=nbin)*weight; events=np.bincount(bins,weights=event[use].astype(float),minlength=nbin)*weight
                    h[0]+=count; h[1]+=events; records[key]+=1
                    eligible_dates[key].add(c['day']); eligible_blocks[key].add(c['anchor'])
                    if events.sum()>0: active_dates[key].add(c['day'])
                    if group=='corridor':
                        perday.setdefault((c['horizon'],cand,c['day']),np.zeros((2,nbin),float))[:] += np.stack((count,events))
            if j%500==0: log(f'{name}: {j}/{len(cases)}')
        calibrators={}; thresholds={}
        for horizon in (1,2):
            for cand in names:
                h=hist.get((horizon,cand,'train','corridor'))
                if h is None: continue
                cal=(np.arange(nbin)+.5)/nbin if probability else old.pava_probabilities(old.Histogram(h[0],h[1]))
                calibrators[horizon,cand]=cal
                vh=hist.get((horizon,cand,'validate','corridor'))
                choices=np.arange(.05,.55,.05)
                def csi(th):
                    pos=cal>=th; hit=vh[1,pos].sum(); miss=vh[1,~pos].sum(); false=(vh[0,pos]-vh[1,pos]).sum()
                    return hit/max(1.,hit+miss+false)
                thresholds[horizon,cand]=float(max(choices,key=csi)) if vh is not None else .2
        rows=[]
        for (horizon,cand,split,group),h in hist.items():
            if (horizon,cand) not in calibrators: continue
            cal=calibrators[horizon,cand]; m=histmetrics(h,cal,thresholds[horizon,cand])
            active=sum(v[1].sum()>0 for (hh,cc,d),v in perday.items() if hh==horizon and cc==cand and self.man['splits'].get(d)==split)
            key=(horizon,cand,split,group)
            row={'experiment':name,'horizon':horizon,'candidate':cand,'split':split,'group':group,'records':records[key],'active_corridor_days':active,'active_group_days':len(active_dates[key]),'eligible_days':len(eligible_dates[key]),'distinct_observation_blocks':len(eligible_blocks[key]),**m}
            train=hist.get((horizon,cand,'train',group),hist.get((horizon,cand,'train','corridor')))
            p=train[1].sum()/train[0].sum(); clim=(h[1].sum()*(1-p)**2+(h[0].sum()-h[1].sum())*p*p)/h[0].sum()
            row['training_climatology_brier']=clim; row['skill_vs_training_climatology']=1-row['brier']/clim if clim>0 else np.nan
            rows.append(row)
        self.results.extend(rows)
        bundle={'hist':hist,'perday':perday,'calibrators':calibrators,'thresholds':thresholds,'names':names,'cases':len(cases)}
        self.bundles[name]=bundle
        return rows
    def choose(self,experiment,reference,names):
        choices={}
        for horizon in (1,2):
            rs=[r for r in self.results if r['experiment']==experiment and r['group']=='corridor' and r['split']=='validate' and r['horizon']==horizon]
            ref=next(r for r in rs if r['candidate']==reference)
            eligible=[r for r in rs if r['candidate'] in names and np.isfinite(r['auc']) and r['auc']>=ref['auc']-.02]
            choices[horizon]=min(eligible,key=lambda r:(r['brier'],-r['auc']))['candidate'] if eligible else reference
        self.selection[experiment]=choices; return choices


def histmetrics(h,p,threshold=.2):
    n,e=h; total=n.sum()
    if total<=0: return {'brier':np.nan,'auc':np.nan,'event_frequency':np.nan}
    b=float(np.sum(e*(1-p)**2+(n-e)*p*p)/total)
    auc=old.auc_from_histogram(old.Histogram(n,e)); pos=e.sum(); neg=(n-e).sum()
    cum_e=np.cumsum(e[::-1]); cum_n=np.cumsum(n[::-1]); precision=np.divide(cum_e,cum_n,out=np.zeros_like(cum_e),where=cum_n>0)
    ap=float(np.sum(e[::-1]*precision)/pos) if pos>0 else np.nan
    yes=p>=threshold; hit=e[yes].sum(); false=(n[yes]-e[yes]).sum(); miss=e[~yes].sum()
    return {'weighted_cells':float(total),'event_frequency':float(pos/total),'brier':b,'auc':auc,'average_precision':ap,'log_loss':float(-np.sum(e*np.log(np.clip(p,1e-6,1-1e-6))+(n-e)*np.log(np.clip(1-p,1e-6,1-1e-6)))/total),'threshold_probability':threshold,'hits_weighted':float(hit),'misses_weighted':float(miss),'false_alarms_weighted':float(false),'probability_of_detection':float(hit/pos) if pos else np.nan,'false_alarm_ratio':float(false/(hit+false)) if hit+false else np.nan,'csi':float(hit/(hit+false+miss)) if hit+false+miss else np.nan}


def run_core(st):
    cases=st.basecases()
    st.evaluate('benchmark',cases,['archive','display_10km'],lambda c:{'archive':(st.scores[c['record']['index'],0],st.targets[c['obs'],2]),'display_10km':(st.scores[c['record']['index'],2],st.targets[c['obs'],2])},True)
    paircases=[]
    for c in cases:
        r=c['record']
        if r['horizon']!=1: continue
        other=st.rec_by.get(lead_pair(r['run'],r['fhour']))
        if other and other['split']==r['split'] and other['valid']==r['valid']:
            paircases.extend([{**c,'peer':other['index']},{**c,'record':other,'horizon':2,'peer':r['index']}])
    names=[f'sigma{s:g}_r{radius:g}' for radius in RADII for s in SCALES]
    def pairedget(c):
        good=np.isfinite(st.scores[c['peer']]).all(0)
        return {f'sigma{s:g}_r{radius:g}':(np.where(good,st.scores[c['record']['index'],k],np.nan),st.targets[c['obs'],ri]) for ri,radius in enumerate(RADII) for k,s in enumerate(SCALES)}
    st.evaluate('spatial_paired',paircases,names,pairedget)
    spatial=st.choose('spatial_paired','sigma10_r30',[f'sigma{s:g}_r30' for s in SCALES])
    # One scale shared by both horizons; validation mean Brier, AUC constraint in each.
    eligible=[]
    for s in SCALES:
        rows=[r for r in st.results if r['experiment']=='spatial_paired' and r['candidate']==f'sigma{s:g}_r30' and r['split']=='validate']
        refs=[r for r in st.results if r['experiment']=='spatial_paired' and r['candidate']=='sigma10_r30' and r['split']=='validate']
        if len(rows)==2 and all(rows[k]['auc']>=refs[k]['auc']-.02 for k in range(2)): eligible.append((np.mean([r['brier'] for r in rows]),s))
    shared=min(eligible)[1] if eligible else 10.
    st.selection['shared_smoothing_km']=shared
    lagcases=[]
    for c in cases:
        r=c['record']; v=dt.datetime.fromisoformat(r['valid'].replace('Z','+00:00'))
        observations=[st.obs_by.get(iso(lag_end(v,l))) for l in LAGS]
        if r['fhour']<9 or r['fhour']>42 or any(o is None for o in observations): continue
        if any(st.man['splits'].get(day(lag_end(v,l)))!=c['split'] for l in LAGS): continue
        lagcases.append({**c,'lag_obs':observations})
    lnames=[f'lag{l:+d}' for l in LAGS]
    st.evaluate('lag_diagnostic',lagcases,lnames,lambda c:{f'lag{l:+d}':(st.scores[c['record']['index'],2],st.targets[c['lag_obs'][k],2]) for k,l in enumerate(LAGS)},True)
    controls=[]
    for c in lagcases:
        v=dt.datetime.fromisoformat(c['anchor'].replace('Z','+00:00'))+dt.timedelta(days=7)
        oi=st.obs_by.get(iso(v))
        if oi is not None and st.man['splits'].get(day(v))==c['split']: controls.append({**c,'control_obs':oi})
    st.evaluate('separated_day_control',controls,['same_day','seven_days_later'],lambda c:{'same_day':(st.scores[c['record']['index'],2],st.targets[c['obs'],2]),'seven_days_later':(st.scores[c['record']['index'],2],st.targets[c['control_obs'],2])},True)
    st.paircases=paircases
    opcases=[]
    for c in cases:
        r=c['record']; prev=[st.rec_by.get((r['run'],r['fhour']-h)) for h in (0,3,6)]
        if r['fhour']<9 or any(p is None or p['version']!=VERSION for p in prev): continue
        opcases.append({**c,'previous':[p['index'] for p in prev]})
    def opget(c):
        a=[st.scores[i,2].astype(np.float32) for i in c['previous']]; y=st.targets[c['obs'],2]
        return {'delay0':(a[0],y),'delay3':(a[1],y),'delay6':(a[2],y),'blend_0_3':((a[0]+a[1])*.5,y)}
    st.evaluate('timing_operational',opcases,['delay0','delay3','delay6','blend_0_3'],opget,True)
    timing=st.choose('timing_operational','delay0',['delay0','delay3','delay6','blend_0_3'])
    def combined(c):
        h=c['horizon']; inds=c['previous']; y=st.targets[c['obs'],2]
        hs=float(spatial[h].split('_')[0][5:]); sk=SCALES.index(hs); sharedk=SCALES.index(shared)
        choice=timing[h]
        selected=(st.scores[inds[0],sk].astype(float)+st.scores[inds[1],sk].astype(float))*.5 if choice=='blend_0_3' else st.scores[inds[int(choice[-1])//3],sk]
        return {'reference':(st.scores[inds[0],2],y),'shared_smoothing':(st.scores[inds[0],sharedk],y),'horizon_smoothing':(st.scores[inds[0],sk],y),'timing_and_smoothing':(selected,y)}
    st.evaluate('combined',opcases,['reference','shared_smoothing','horizon_smoothing','timing_and_smoothing'],combined)
    st.choose('combined','reference',['reference','shared_smoothing','horizon_smoothing','timing_and_smoothing'])
    six=[]
    for c in cases:
        r=c['record']; following=st.rec_by.get((r['run'],r['fhour']+3))
        if following and following['valid'] in st.obs_by and following['split']==c['split']:
            six.append({**c,'next_record':following['index'],'next_obs':st.obs_by[following['valid']]})
    def sixget(c):
        y=st.targets[c['obs'],2]; yy=st.targets[c['next_obs'],2]; event=np.where((y==255)|(yy==255),255,np.maximum(y,yy))
        a=st.scores[c['record']['index'],2]; b=st.scores[c['next_record'],2]
        return {'current_LPI':(a,event),'max_two_forecast_blocks':(np.maximum(a,b),event)}
    st.evaluate('six_hour_target',six,['current_LPI','max_two_forecast_blocks'],sixget)
    st.selection['counts']={'baseline_records':len(cases),'paired_records_per_horizon':len(paircases)//2,'lag_common_records':len(lagcases),'operational_common_records':len(opcases),'six_hour_records':len(six)}
    log('Core experiments completed')


def run_neighborhood(st):
    """Fixed LPI20 threshold; Gaussian fractions skill, diagnostic not tuning."""
    selected={}
    for c in st.paircases:
        key=c['anchor'],c['horizon']
        if key not in selected: selected[key]=c
    accum={}; perday=[]
    for c in selected.values():
        if c['split']=='train': continue
        score=st.scores[c['record']['index'],2]; yy=st.targets[c['obs'],0]
        good=np.isfinite(score)&np.isfinite(st.scores[c['peer'],2])&(yy!=255)
        shape=st.grid['lat'].shape; weight=np.zeros(shape,np.float32); f=np.zeros(shape,np.float32); o=np.zeros(shape,np.float32)
        ids=st.grid['indices'][good]; weight.ravel()[ids]=1; f.ravel()[ids]=(score[good]>=20); o.ravel()[ids]=(yy[good]==1)
        common_weight=gaussian_filter(weight,40/5,mode='constant',truncate=4)
        common_use=(common_weight.ravel()[st.grid['indices']]>=.95)&st.cor&good
        for km in (0.,10.,20.,40.):
            w=gaussian_filter(weight,km/5,mode='constant',truncate=4) if km else weight
            pf=gaussian_filter(f,km/5,mode='constant',truncate=4) if km else f
            po=gaussian_filter(o,km/5,mode='constant',truncate=4) if km else o
            use=common_use; i=st.grid['indices'][use]
            if not len(i): continue
            a=pf.ravel()[i]/w.ravel()[i]; b=po.ravel()[i]/w.ravel()[i]
            num=float(np.sum((a-b)**2)); den=float(np.sum(a*a+b*b)); key=c['horizon'],c['split'],km
            values=accum.setdefault(key,[0.,0.,0,0]); values[0]+=num; values[1]+=den; values[2]+=len(i); values[3]+=1
            perday.append({'horizon':c['horizon'],'split':c['split'],'day':c['day'],'valid':c['anchor'],'neighborhood_sigma_km':km,'squared_error':num,'squared_reference':den,'cells':len(i)})
    rows=[{'horizon':h,'split':s,'neighborhood_sigma_km':k,'fss':1-v[0]/v[1] if v[1] else np.nan,'cells':v[2],'blocks':v[3]} for (h,s,k),v in accum.items()]
    write_csv(st.out/'neighborhood_fss.csv',rows); write_csv(st.out/'neighborhood_fss_blocks.csv',perday); st.fss=rows
    log('Neighborhood displacement diagnostic completed')


def bootstrap(bundle,horizon,cand,ref,dates,block=1,draws=2000):
    pc=bundle['calibrators'][horizon,cand]; pr=bundle['calibrators'][horizon,ref]; days=sorted(d for d in dates if (horizon,cand,d) in bundle['perday'] and (horizon,ref,d) in bundle['perday'])
    if len(days)<3: return {'days':len(days),'brier_low':np.nan,'brier_high':np.nan,'auc_low':np.nan,'auc_high':np.nan}
    ah=np.stack([bundle['perday'][horizon,cand,d] for d in days]); bh=np.stack([bundle['perday'][horizon,ref,d] for d in days]); rng=np.random.default_rng(SEED+block)
    diffs=[]
    for _ in range(draws):
        if block==1: ids=rng.integers(0,len(days),len(days))
        else:
            starts=rng.integers(0,len(days),math.ceil(len(days)/block)); ids=np.concatenate([(np.arange(s,s+block)%len(days)) for s in starts])[:len(days)]
        a=histmetrics(ah[ids].sum(0),pc); b=histmetrics(bh[ids].sum(0),pr); diffs.append((a['brier']-b['brier'],a['auc']-b['auc']))
    vals=np.asarray(diffs)
    return {'days':len(days),'brier_low':float(np.nanpercentile(vals[:,0],2.5)),'brier_high':float(np.nanpercentile(vals[:,0],97.5)),'auc_low':float(np.nanpercentile(vals[:,1],2.5)),'auc_high':float(np.nanpercentile(vals[:,1],97.5))}


def finalize_statistics(st):
    rows=[]; development=list(getattr(st,'solar_rolling',[]))
    tests=[d for d,s in st.man['splits'].items() if s=='test']
    configs=[('spatial_paired','sigma10_r30',lambda h:[f'sigma{s:g}_r30' for s in SCALES]),('timing_operational','delay0',lambda h:['delay0','delay3','delay6','blend_0_3']),('combined','reference',lambda h:st.bundles['combined']['names'])]
    if 'ingredients' in st.bundles: configs.append(('ingredients','recomputed_baseline',lambda h:st.bundles['ingredients']['names']))
    if 'solar' in st.bundles: configs.append(('solar','lpi_time_horizon',lambda h:st.bundles['solar']['names']))
    dates=st.man['dates']; devdates=[d for d in dates if st.man['splits'].get(d)!='test']
    for exp,ref,func in configs:
        bundle=st.bundles[exp]
        for h in (1,2):
            if (h,ref) not in bundle['calibrators']: continue
            for cand in func(h):
                if (h,cand) not in bundle['calibrators']: continue
                for block in (1,3): rows.append({'experiment':exp,'horizon':h,'candidate':cand,'reference':ref,'resampling_block_days':block,**bootstrap(bundle,h,cand,ref,tests,block)})
                for f,frac in enumerate((.50,.75),1):
                    b=dt.date.fromisoformat(devdates[int(len(devdates)*frac)])
                    end=dt.date.fromisoformat(devdates[min(len(devdates)-1,int(len(devdates)*(frac+.20)))])
                    tr=[d for d in devdates if dt.date.fromisoformat(d)<b-dt.timedelta(days=3)]
                    va=[d for d in devdates if b+dt.timedelta(days=3)<=dt.date.fromisoformat(d)<=end]
                    def metric(cc):
                        a=[bundle['perday'][h,cc,d] for d in tr if (h,cc,d) in bundle['perday']]; vv=[bundle['perday'][h,cc,d] for d in va if (h,cc,d) in bundle['perday']]
                        if not a or not vv: return None
                        ah=np.sum(a,axis=0); vh=np.sum(vv,axis=0)
                        if exp=='solar': return None # Models require actual refitting in each fold.
                        p=old.pava_probabilities(old.Histogram(ah[0],ah[1])); return histmetrics(vh,p)
                    mm=metric(cand); rr=metric(ref)
                    if mm and rr: development.append({'experiment':exp,'horizon':h,'candidate':cand,'fold':f,'train_days':len(tr),'validation_days':len(va),'brier_delta':mm['brier']-rr['brier'],'auc_delta':mm['auc']-rr['auc']})
    write_csv(st.out/'bootstrap_differences.csv',rows); write_csv(st.out/'rolling_development.csv',development)
    st.bootstrap=rows; st.development=development


def solar_design(st,c,score,cal):
    sol=st.solar(c['anchor']); bins=np.clip((np.nan_to_num(score)*2).astype(int),0,199)
    baseline=logit(np.clip(cal[bins],1e-6,1-1e-6)).astype(np.float64)
    baseline[~np.isfinite(score)]=np.nan
    return np.column_stack((np.ones(len(score)),baseline,sol['sin'],sol['cos'],np.full(len(score),c['horizon']-1),np.sin(np.deg2rad(sol['elevation'])),np.full(len(score),sol['declination']/.4),(sol['daylength']-12)/6,np.full(len(score),sol['calendar'])))


def fit_logistic(x,n,e,columns):
    xx=x[:,columns]; w=n/n.sum(); y=e/n
    def objective(beta):
        z=xx@beta; p=expit(z); reg=.0002
        loss=np.sum(w*(np.logaddexp(0,z)-y*z))+reg*np.sum(beta[1:]**2)
        grad=xx.T@(w*(p-y)); grad[1:]+=2*reg*beta[1:]
        return loss,grad
    bounds=[(None,None)]*len(columns)
    if 1 in columns: bounds[columns.index(1)]=(0,None)
    beta=np.zeros(len(columns)); beta[0]=logit(np.clip(e.sum()/n.sum(),1e-5,1-1e-5))
    result=minimize(objective,beta,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':150,'ftol':1e-10})
    if not result.success: log(f'Logistic optimizer note: {result.message}')
    return result.x,{'converged':bool(result.success),'message':str(result.message),'iterations':int(result.nit)}


def run_solar(st):
    cases=st.basecases(); mult=collections.Counter((c['anchor'],c['horizon']) for c in cases)
    # Count-weighted sufficient groups by score, solar-elevation bin and local-time bin.
    rows=[]; ns=[]; es=[]; splits=[]; days=[]; rawbins=[]; horizons=[]
    reference=st.bundles['benchmark']['calibrators']
    for j,c in enumerate(cases):
        score=st.scores[c['record']['index'],2].astype(float); event=st.targets[c['obs'],2]
        use=st.cor&(event!=255)&np.isfinite(score)
        x=solar_design(st,c,score,reference[c['horizon'],'display_10km']); sol=st.solar(c['anchor'])
        group=(np.clip((np.nan_to_num(score)*2).astype(int),0,199)*16+np.clip(((sol['elevation']+90)/15).astype(int),0,15))*8+np.clip((sol['hour']/3).astype(int),0,7)
        ids,inv=np.unique(group[use],return_inverse=True); n=np.bincount(inv); e=np.bincount(inv,weights=event[use]); xx=np.column_stack([np.bincount(inv,weights=x[use,k])/n for k in range(x.shape[1])])
        rows.append(xx); ns.append(n/mult[c['anchor'],c['horizon']]); es.append(e/mult[c['anchor'],c['horizon']]); splits.extend([c['split']]*len(n)); days.extend([c['day']]*len(n))
        rawbins.extend((ids//128).tolist()); horizons.extend([c['horizon']]*len(n))
        if j%700==0: log(f'Solar grouped fitting table {j}/{len(cases)}')
    x=np.concatenate(rows); n=np.concatenate(ns); e=np.concatenate(es); splits=np.asarray(splits); days=np.asarray(days); rawbins=np.asarray(rawbins); horizons=np.asarray(horizons)
    models={'constant':[0],'time_horizon_climatology':[0,2,3,4],'lpi_only':[0,1],'lpi_time_horizon':[0,1,2,3,4],'lpi_solar_elevation':[0,1,2,3,4,5],'lpi_declination':[0,1,2,3,4,6],'lpi_daylength':[0,1,2,3,4,7],'lpi_calendar':[0,1,2,3,4,8],'solar_only':[0,2,3,4,5,6]}
    params={}; notes={}; train=splits=='train'
    for name,columns in models.items():
        beta,info=fit_logistic(x[train],n[train],e[train],columns); params[name]=beta; notes[name]={'columns':columns,'coefficients':beta.tolist(),**info}
    def get(c):
        score=st.scores[c['record']['index'],2].astype(float); xx=solar_design(st,c,score,reference[c['horizon'],'display_10km']); event=st.targets[c['obs'],2]
        return {name:(expit(xx[:,columns]@params[name]),event) for name,columns in models.items()}
    st.evaluate('solar',cases,list(models),get,True,True)
    st.choose('solar','lpi_time_horizon',['lpi_time_horizon','lpi_solar_elevation','lpi_declination','lpi_daylength','lpi_calendar'])
    # Refit low-complexity extensions on rolling development dates; primary train PAVA
    # logit feature would leak for earlier folds, so refit that mapping too.
    rolling=[]; devdates=[d for d in st.man['dates'] if st.man['splits'].get(d)!='test']
    for fold,frac in enumerate((.5,.75),1):
        boundary=dt.date.fromisoformat(devdates[int(len(devdates)*frac)]); end=devdates[min(len(devdates)-1,int(len(devdates)*(frac+.20)))];
        tr=np.array([dt.date.fromisoformat(d)<boundary-dt.timedelta(days=3) for d in days]); va=np.array([boundary+dt.timedelta(days=3)<=dt.date.fromisoformat(d)<=dt.date.fromisoformat(end) for d in days])
        if tr.sum() and va.sum():
            fold_scores={}; xx=x.copy(); foldcal={}
            for h in (1,2):
                use=tr&(horizons==h)
                counts=np.bincount(rawbins[use],weights=n[use],minlength=200); events=np.bincount(rawbins[use],weights=e[use],minlength=200)
                cal=old.pava_probabilities(old.Histogram(counts,events)); foldcal[h]=cal
                xx[horizons==h,1]=logit(np.clip(cal[rawbins[horizons==h]],1e-6,1-1e-6))
            for name in ('lpi_time_horizon','lpi_solar_elevation','lpi_declination','lpi_daylength','lpi_calendar'):
                cols=models[name]; beta,info=fit_logistic(xx[tr],n[tr],e[tr],cols)
                hh={h:np.zeros((2,1000)) for h in (1,2)}
                for c in cases:
                    d=dt.date.fromisoformat(c['day'])
                    if not (boundary+dt.timedelta(days=3)<=d<=dt.date.fromisoformat(end)): continue
                    score=st.scores[c['record']['index'],2].astype(float); event=st.targets[c['obs'],2]; use=st.cor&(event!=255)&np.isfinite(score)
                    pred=expit(solar_design(st,c,score,foldcal[c['horizon']])[:,cols]@beta); bins=np.clip((pred[use]*1000).astype(int),0,999)
                    weight=1./mult[c['anchor'],c['horizon']]
                    hh[c['horizon']][0]+=np.bincount(bins,minlength=1000)*weight; hh[c['horizon']][1]+=np.bincount(bins,weights=event[use],minlength=1000)*weight
                for h in (1,2): fold_scores[h,name]=histmetrics(hh[h],(np.arange(1000)+.5)/1000)
            for (h,name),val in fold_scores.items():
                ref=fold_scores[h,'lpi_time_horizon']; rolling.append({'experiment':'solar','horizon':h,'fold':fold,'candidate':name,'train_days':len(set(days[tr])),'validation_days':len(set(days[va])),'brier_delta':val['brier']-ref['brier'],'auc_delta':val['auc']-ref['auc'],'note':'PAVA and logistic models independently refitted; exact pixel evaluation'})
    st.solar_rolling=rolling
    write_csv(st.out/'solar_development_sensitivity.csv',rolling)
    write_json(st.out/'solar_models.json',notes)
    st.solar_notes=notes
    # Exact evaluated-pixel solar residual summaries, including day length and declination.
    residual=[]
    for c in cases:
        score=st.scores[c['record']['index'],2].astype(float); y=st.targets[c['obs'],2]; cal=reference[c['horizon'],'display_10km']; p=cal[np.clip((score*2).astype(int),0,199)]; sol=st.solar(c['anchor'])
        for regime,m in [('night',sol['elevation']<-6),('twilight',(sol['elevation']>=-6)&(sol['elevation']<0)),('day_low',(sol['elevation']>=0)&(sol['elevation']<30)),('day_high',sol['elevation']>=30)]:
            use=st.cor&m&(y!=255)&np.isfinite(score)
            if use.any(): residual.append({'valid':c['anchor'],'day':c['day'],'horizon':c['horizon'],'split':c['split'],'month':c['anchor'][5:7],'regime':regime,'cells':int(use.sum()),'observed':float(y[use].mean()),'forecast':float(p[use].mean()),'residual':float(y[use].mean()-p[use].mean()),'daylength':float(sol['daylength'][use].mean()),'maximum_elevation':float(sol['maximum'][use].mean()),'declination_degrees':float(np.rad2deg(sol['declination'])),'daylight_fraction':float(sol['daylight'][use].mean())})
    write_csv(st.out/'solar_residuals.csv',residual)
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for regime in ('night','twilight','day_low','day_high'):
        rows=[r for r in residual if r['regime']==regime]; months=sorted(set(r['month'] for r in rows)); values=[]
        for month in months:
            rr=[r for r in rows if r['month']==month]
            values.append(np.average([r['residual'] for r in rr],weights=[r['cells'] for r in rr]))
        ax.plot(months,values,'-o',label=regime)
    ax.axhline(0,color='gray',linestyle='--'); ax.set(xlabel='Verifying month (2026)',ylabel='Observed minus calibrated occurrence',title='Solar-regime calibration residuals (descriptive)'); ax.legend(); fig.savefig(st.out/'solar_residuals.png',dpi=160); plt.close(fig)
    log('Solar experiments completed')


def run_ingredients(st,root):
    hourly=archive.hourly_lpi_archive_dir(root); cor_indices=st.grid['indices'][st.cor]
    formulas=old.candidate_formulas(); names=[f.name for f in formulas]; lookup={}
    for c in st.basecases():
        r=c['record']; rd=hourly/r['run'][:4]/r['run']; mp=rd/'manifest.json'
        if not mp.exists() or not json.loads(mp.read_text()).get('complete'): continue
        paths=[rd/f'f{h:03d}.npz' for h in range(r['fhour']-2,r['fhour']+1)]
        if not all(p.exists() for p in paths): continue
        lookup[r['index']]=(c,paths)
    cases=[c for c,paths in lookup.values()]
    cache={}; sources=[]; cond=[]
    def get(c):
        maxima={name:np.full(len(cor_indices),-np.inf,np.float32) for name in names}
        issued=np.full(len(cor_indices),-np.inf,np.float32); means=collections.defaultdict(list)
        for p in lookup[c['record']['index']][1]:
            side=json.loads(p.with_suffix('.json').read_text()); sha=digest(p)
            if sha!=side['sha256']: raise RuntimeError(f'Ingredient checksum mismatch {p}')
            with np.load(p) as z:
                if str(z['formula_version'].item())!=VERSION: return None
                if str(z['grid_hash'].item())!=st.man['grid_hash']: raise RuntimeError('Ingredient grid mismatch')
            fields=old.read_hour_fields(p,cor_indices)
            for k,v in fields.items(): means[k].append(float(np.nanmean(v)))
            for f in formulas: maxima[f.name]=np.maximum(maxima[f.name],old.compute_formula(fields,f))
            issued=np.maximum(issued,fields['potential'])
            sources.append({'path':str(p),'sha256':sha})
        # Compare reconstructed formulas on identical spatial processing. Exact issued
        # potential is additional diagnostic, not the formula-selection reference.
        y=st.targets[c['obs'],2]; result={}
        for name,a in {**maxima,'issued_hourly_max':issued}.items():
            full=np.full(len(st.cor),np.nan,np.float32); full[st.cor]=a; result[name]=(full,y)
        cond.append({'valid':c['anchor'],'run':c['record']['run'],'horizon':c['horizon'],'split':c['split'],**{k:float(np.mean(v)) for k,v in means.items()}})
        return result
    st.evaluate('ingredients',cases,names+['issued_hourly_max'],get)
    st.choose('ingredients','recomputed_baseline',names)
    write_csv(st.out/'ingredient_diagnostics.csv',cond); write_csv(st.out/'ingredient_sources.csv',list({r['path']:r for r in sources}.values()))
    st.selection['ingredient_cases']=len(cases)


def plots_and_cases(st):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for h,ax in enumerate(axes,1):
        for split in ('validate','test'):
            rs=[r for r in st.results if r['experiment']=='lag_diagnostic' and r['horizon']==h and r['group']=='corridor' and r['split']==split]
            ax.plot(LAGS,[next(r['brier'] for r in rs if r['candidate']==f'lag{l:+d}') for l in LAGS],'-o',label=split)
        ax.set(title=f'Day {h}: lightning timing',xlabel='Observation offset after LPI (hours)',ylabel='Brier score (lower is better)'); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(st.out/'timing_lags.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for h,ax in enumerate(axes,1):
        for split in ('validate','test'):
            rs=[r for r in st.results if r['experiment']=='spatial_paired' and r['horizon']==h and r['split']==split]
            ax.plot(SCALES,[next(r['brier'] for r in rs if r['candidate']==f'sigma{s:g}_r30') for s in SCALES],'-o',label=split)
        ax.set(title=f'Day {h}: matched storms, 30 km target',xlabel='Gaussian sigma (km)',ylabel='Brier score'); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(st.out/'spatial_smoothing.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for h,ax in enumerate(axes,1):
        b=st.bundles['benchmark']
        for name in ('archive','display_10km'):
            hist=b['hist'][h,name,'test','corridor']; p=b['calibrators'][h,name]; bins=np.clip((p*10).astype(int),0,9); xx=[]; yy=[]
            for k in range(10):
                use=bins==k; n=hist[0,use].sum()
                if n: xx.append(np.sum(p[use]*hist[0,use])/n); yy.append(hist[1,use].sum()/n)
            ax.plot(xx,yy,'-o',label=name)
        ax.plot([0,1],[0,1],'--',color='gray'); ax.set(title=f'Day {h}: test reliability',xlabel='Forecast probability',ylabel='Observed frequency'); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(st.out/'reliability.png',dpi=160); plt.close(fig)
    rs=[r for r in st.results if r['experiment']=='solar' and r['group']=='corridor' and r['split']=='test']
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for h,ax in enumerate(axes,1):
        rr=[r for r in rs if r['horizon']==h]; ax.barh([r['candidate'].replace('lpi_','') for r in rr],[r['brier'] for r in rr]); ax.set(title=f'Day {h}: solar and seasonal models',xlabel='Test Brier score')
    fig.savefig(st.out/'solar_models.png',dpi=160); plt.close(fig)
    # Validation cases chosen only by observed intensity; they do not select candidates.
    cases=[c for c in st.basecases() if c['split']=='validate' and c['horizon']==1]
    case_by_time={c['anchor']:c for c in cases}; ordered=sorted(case_by_time.values(),key=lambda c:float(np.nanmean(st.density[c['obs'],st.cor])))
    case_rows=[]; detail=[]
    for title,quantile in [('quiet',.05),('marginal',.50),('organized',.95)]:
        c=ordered[min(len(ordered)-1,int((len(ordered)-1)*quantile))]; idx=c['record']['index']; h=c['horizon']; selected=float(st.selection['spatial_paired'][h].split('_')[0][5:]); sk=SCALES.index(selected)
        arrays=[st.density[c['obs']].astype(float),st.scores[idx,0],st.scores[idx,2],st.scores[idx,sk]]
        titles=['Observed flash density','Archived LPI','Current display: sigma 10 km',f'Validation choice: sigma {selected:g} km']
        fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
        for ax,a,labeltext in zip(axes.flat,arrays,titles):
            grid=np.full(st.grid['lat'].shape,np.nan); grid.ravel()[st.grid['indices']]=a
            vmax=max(.01,float(np.nanpercentile(a,99))) if labeltext.startswith('Observed') else 100
            plot=ax.pcolormesh(st.grid['lon'],st.grid['lat'],grid,cmap='magma',vmin=0,vmax=vmax,shading='auto'); ax.set(title=labeltext,xlabel='Longitude',ylabel='Latitude',xlim=(-139,-114),ylim=(48,60)); fig.colorbar(plot,ax=ax,shrink=.7)
        fig.suptitle(f'{title.title()} validation case: {c["anchor"]}, {c["record"]["run"]}')
        path=f'case_{title}.png'; fig.savefig(st.out/path,dpi=145); plt.close(fig)
        case_rows.append({'kind':title,'valid':c['anchor'],'run':c['record']['run'],'fhour':c['record']['fhour'],'file':path})
        for k,s in enumerate(SCALES):
            a=st.scores[idx,k]; grid=np.full(st.grid['lat'].shape,np.nan); grid.ravel()[st.grid['indices']]=a
            mask=np.isfinite(grid); dx=np.abs(np.diff(grid,axis=1)); dy=np.abs(np.diff(grid,axis=0)); _,components=label(np.nan_to_num(grid)>=20,structure=np.ones((3,3)))
            detail.append({'case':title,'sigma_km':s,'area_fraction_ge20':float(np.mean(a>=20)),'mean_neighbor_difference':float(np.nanmean(np.concatenate((dx.ravel(),dy.ravel())))),'components_ge20':int(components)})
    write_csv(st.out/'case_index.csv',case_rows); write_csv(st.out/'spatial_detail.csv',detail)
    # All complete run sequences: first modeled high LPI and first observed event.
    byrun=collections.defaultdict(dict)
    for c in st.basecases(): byrun[c['record']['run']][c['record']['fhour']]=c
    onset=[]; onset_counts=collections.Counter(); sequence_examples=[]
    for run,cs in byrun.items():
        if any(h not in cs for h in range(3,49,3)): continue
        pp=np.stack([st.scores[cs[h]['record']['index'],2][st.cor]>=20 for h in range(3,49,3)])
        yy=np.stack([st.targets[cs[h]['obs'],2][st.cor] for h in range(3,49,3)])
        use=(yy!=255).all(0)&pp.any(0)&(yy==1).any(0)
        pi=pp.argmax(0); yi=(yy==1).argmax(0); use &= (pi>0)&(yi>0)
        offsets=(yi[use]-pi[use])*3
        for delta,n in collections.Counter(offsets.tolist()).items():
            onset.append({'run':run,'day':cs[3]['day'],'offset_hours':delta,'cells':n,'eligible_cells_run':len(offsets),'fraction_run':n/len(offsets) if len(offsets) else 0}); onset_counts[delta]+=n
        if len(offsets): sequence_examples.append((float(np.abs(offsets).mean()),run,cs))
    write_csv(st.out/'first_event_offsets.csv',onset)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    ks=sorted(onset_counts); ax.bar(ks,[onset_counts[k] for k in ks],width=2.5); ax.set(xlabel='First observed active block minus first LPI ≥20 block (hours)',ylabel='Corridor cell sequences (correlated)',title='Onset diagnostic: complete runs, censored starts excluded'); fig.savefig(st.out/'onset_offsets.png',dpi=160); plt.close(fig)
    if sequence_examples:
        _,run,cs=max(sequence_examples)
        fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True); hs=list(range(3,49,3))
        ax.plot(hs,[float(np.mean(st.scores[cs[h]['record']['index'],2][st.cor])) for h in hs],'-o',label='Mean displayed LPI'); ax.set(xlabel='Forecast lead (hours)',ylabel='Mean LPI',title=f'Timing diagnostic case {run}')
        ax2=ax.twinx(); ax2.plot(hs,[float(np.mean(st.targets[cs[h]['obs'],2][st.cor]==1)) for h in hs],'-s',color='darkorange',label='Observed 30 km occurrence fraction'); ax2.set_ylabel('Observed corridor occurrence fraction'); ax.legend(loc='upper left'); ax2.legend(loc='upper right'); fig.savefig(st.out/'timing_case.png',dpi=160); plt.close(fig)
    st.selection['onset_complete_runs']=sum(all(h in cs for h in range(3,49,3)) for cs in byrun.values())


def report(st):
    write_csv(st.out/'metrics.csv',st.results); write_json(st.out/'selection.json',st.selection)
    def table(rows,columns):
        def fmt(v):
            if isinstance(v,float): return f'{v:.4f}' if math.isfinite(v) else 'unavailable'
            return html.escape(str(v))
        return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(k.replace('_',' '))+'</th>' for k in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+fmt(r.get(k,''))+'</td>' for k in columns)+'</tr>' for r in rows)+'</tbody></table></div>'
    primary=[r for r in st.results if r['group']=='corridor' and r['split']=='test']
    baseline=[r for r in primary if r['experiment']=='benchmark']
    selected={e:cs for e,cs in st.selection.items() if isinstance(cs,dict) and 1 in cs}
    conclusions=[]; gate_rows=[]
    for e,ref in [('spatial_paired','sigma10_r30'),('timing_operational','delay0'),('combined','reference'),('ingredients','recomputed_baseline'),('solar','lpi_time_horizon')]:
        if e not in selected: continue
        for h,cand in selected[e].items():
            if not isinstance(h,int): continue
            c=next((r for r in primary if r['experiment']==e and r['horizon']==h and r['candidate']==cand),None); b=next((r for r in primary if r['experiment']==e and r['horizon']==h and r['candidate']==ref),None)
            boot=next((r for r in st.bootstrap if r['experiment']==e and r['horizon']==h and r['candidate']==cand and r['resampling_block_days']==1),{})
            rolls=[r for r in st.development if r['experiment']==e and r['horizon']==h and r['candidate']==cand]
            repeat=sum(r['brier_delta']<0 and r['auc_delta']>=-.02 for r in rolls)
            passed=bool(c and b and cand!=ref and c['auc']>=b['auc']-.02 and boot.get('brier_high',float('inf'))<0 and repeat>=2)
            gate_rows.append({'experiment':e,'horizon':h,'validation_choice':cand,'test_brier_delta':c['brier']-b['brier'] if c and b else np.nan,'test_auc_delta':c['auc']-b['auc'] if c and b else np.nan,'bootstrap_brier_low':boot.get('brier_low',np.nan),'bootstrap_brier_high':boot.get('brier_high',np.nan),'development_folds_improved':repeat,'trial_gate_passed':passed})
    write_csv(st.out/'decision_gates.csv',gate_rows)
    passed=[r for r in gate_rows if r['trial_gate_passed']]
    decision='Candidates meet the statistical trial gate; inspect native-grid reproduction before proposing a parallel trial.' if passed else 'Retain current operational guidance. No selected change met every predeclared trial requirement in this study.'
    fig=lambda name,caption:'<figure><img src="'+name+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'
    parts=['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LPI verification results</title><style>body{font:17px/1.6 system-ui,sans-serif;color:#203140;background:#f3f6f8;margin:0}main{max-width:1120px;margin:auto;background:white;padding:40px}h1{font-size:36px;line-height:1.2}h2{margin-top:40px;color:#173c55}p{max-width:900px}.decision{background:#eaf2f7;padding:20px;border-left:5px solid #2a6388}.table{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #dce3e9}th{background:#edf2f6}figure{margin:25px 0}img{max-width:100%;height:auto}figcaption{font-size:14px;color:#586976}a{color:#225e89}code{font-size:14px}nav a{margin-right:14px}.small{font-size:14px;color:#586976}</style></head><body><main><h1>LPI verification results</h1><p class="small">2026 archive study · prepared October 2026 · forecast and observation inputs frozen before evaluation</p>',f'<p class="decision"><strong>{decision}</strong></p>','<nav><a href="#data">Data</a><a href="#timing">Timing</a><a href="#spatial">Location</a><a href="#solar">Season and sun</a><a href="#ingredients">Ingredients</a><a href="#decision">Decisions</a></nav>']
    man=st.man
    parts+=['<h2 id="data">Data and validation</h2>',f'<p>Observation cutoff: <strong>{man["cutoff"]}</strong>. The common domain contains {man["mask_corridor_cells"]:,} BCH corridor cells and {man["mask_bc_cells"]:,} BC cells. Originally there were {man["original_corridor_cells"]:,} corridor cells. Grid-edge cells without an 80 km margin were excluded. Each smoothed field additionally requires at least 95% finite Gaussian weight and a finite center. Comparisons use the intersection of eligible cells across candidates; paired horizons also share cells. Dynamic missing coverage makes results conditional on available forecast data. The primary event is at least one positive observed flash-density cell center within 30 km during a three-hour interval.</p>',f'<p>Training/validation and validation/test boundaries are {man["boundaries"][0]} and {man["boundaries"][1]}. Dates within one day of a boundary and issued runs spanning folds are excluded. All tests share the final chronological dates. Repeated cycles verifying one block are down-weighted. Observational distance is measured in EPSG:3005; nodata within the event radius plus 3 km causes exclusion. Targets use gridded flash-cell centers, not precise strike locations.</p>',table(baseline,['horizon','candidate','records','active_corridor_days','event_frequency','brier','auc','average_precision','skill_vs_training_climatology']),fig('reliability.png','Probabilities fitted on training dates, evaluated on the untouched final chronological period.')]
    parts+=['<h2 id="timing">Does LPI precede observed lightning?</h2>','<p>Positive offsets compare an LPI block with lightning occurring later. The diagnostic uses the same forecast intervals with all five observed offsets available and holds the target duration and radius fixed. Operational tests instead predict a fixed observed block using the current, previous or earlier forecast block from the same issued run. A blend is the equal-weight mean of current and preceding LPI. Early leads without predecessor fields are excluded from every operational candidate.</p>',fig('timing_lags.png','Positive offset means lightning occurs after the LPI block; negative means lightning precedes it.'),table([r for r in primary if r['experiment']=='timing_operational'],['horizon','candidate','records','brier','auc','average_precision','csi']),fig('onset_offsets.png','First-block timing is descriptive: each run/cell sequence is correlated with other cycles and neighboring cells; ongoing events at the first block are excluded.'),fig('timing_case.png','A complete run selected for visible onset mismatch; this case does not choose a tuning parameter.'),'<p>The six-hour sensitivity predicts occurrence in the union of two adjacent observed blocks. Its event frequency and scores answer a different question from the three-hour study.</p>',table([r for r in primary if r['experiment']=='six_hour_target'],['horizon','candidate','event_frequency','brier','auc'])]
    parts+=['<h2>Timing control</h2>',table([r for r in primary if r['experiment']=='separated_day_control'],['horizon','candidate','records','event_frequency','brier','auc']),'<p>The seven-day control preserves the UTC block and location but verifies a different weather day within the same split. It is a diagnostic of persistence and daily-cycle confounding, with its own event frequency.</p>']
    parts+=['<h2 id="spatial">Does day two need more spatial smoothing?</h2>','<p>The paired comparison verifies the same observed blocks using forecasts from the same issue cycle 24 hours apart. F003 pairs with F027, and so on. Smoothing is Gaussian sigma in kilometres after the three-hour maximum. Current display smoothing is 10 km. All candidates are calibrated independently on training dates; choices are made on validation Brier score subject to AUC staying within 0.02 of the reference.</p>',fig('spatial_smoothing.png','Matched-storm comparison at the fixed 30 km corridor target.'),table([r for r in primary if r['experiment']=='spatial_paired' and r['candidate'].endswith('_r30')],['horizon','candidate','records','brier','auc','average_precision']),'<p>The radius sweep changes the observed event. Larger radii increase event frequency; lower Brier scores across radii do not by themselves mean better location skill. Skill is shown relative to each target’s training climatology.</p>',table([r for r in primary if r['experiment']=='spatial_paired' and r['candidate'].startswith('sigma10_')],['horizon','candidate','event_frequency','brier','auc','skill_vs_training_climatology'])]
    parts+=['<h2>Displacement tolerance</h2>',table(st.fss,['horizon','split','neighborhood_sigma_km','blocks','fss']),'<p>Fractions skill score compares LPI at least 20 with observed occurrence within 10 km. Gaussian neighborhood sigma is varied while the event and LPI threshold stay fixed. One deterministic same-cycle pair per observed block is retained. A value of one is exact agreement; zero is no overlap. This is a location diagnostic, not a calibrated warning threshold.</p>']
    parts+=['<h2 id="solar">Seasonal and solar information</h2>','<p>Solar geometry is calculated from each location and interval midpoint. The nested probability models test whether solar elevation, day length, declination or calendar position add information beyond LPI, local solar time and horizon. LPI enters through a training-fitted monotone probability feature with a nonnegative coefficient. Small regularization limits extra terms. Fitting groups nearby score, solar-elevation and solar-time values into count-weighted bins; evaluation predicts each eligible grid cell directly.</p>',fig('solar_residuals.png','Calibration residuals by solar regime and verifying month; positive means underprediction.'),fig('solar_models.png','Final-period probability scores. Compare extensions with lpi_time_horizon, not only with a constant climatology.'),table([r for r in primary if r['experiment']=='solar'],['horizon','candidate','brier','auc','average_precision']),'<p>This single July-to-October season cannot separate a causal sun-angle effect from seasonal weather or establish an annual correction. Seasonal terms need repeatable chronological performance and another season. Both the LPI probability mapping and logistic model are independently refitted in each rolling development fold.</p>']
    parts+=['<h2 id="ingredients">Ingredient tuning on the available subset</h2>','<p>The full predefined set of constrained formula adjustments and component ablations is evaluated where all three hourly ingredient snapshots exist in a complete run. Formula candidates share the same archive-grid reconstruction, compared with recomputed_baseline. The exact issued hourly maximum is an additional reference. These tests exclude post-window display smoothing to isolate the ingredient formula. Native-grid processing and a full day-two ingredient archive are still required before promoting a formula change.</p>',table([r for r in primary if r['experiment']=='ingredients' and (r['candidate'].startswith('ablate_') or r['candidate'] in ('recomputed_baseline','issued_hourly_max','charge_depth_50_175'))],['horizon','candidate','records','brier','auc','average_precision'])]
    parts+=['<h2>Case maps and location detail</h2>','<p>Quiet, middle-intensity and organized examples are selected by observed intensity on validation dates. The maps show only the common supported verification domain; missing surroundings are blank. They are diagnostic examples rather than the meteorologist-labeling exercise.</p>']
    for name in ('quiet','marginal','organized'): parts.append(fig(f'case_{name}.png',f'{name.title()} validation case'))
    parts+=['<h2 id="decision">Candidate decisions and remaining evidence</h2>',table(gate_rows,['experiment','horizon','validation_choice','test_brier_delta','test_auc_delta','bootstrap_brier_low','bootstrap_brier_high','development_folds_improved','trial_gate_passed']),'<p>The interval is candidate minus reference, resampled by verifying day with 2,000 draws. Negative Brier differences favor the candidate. Three-day circular blocks of consecutive available verifying dates are also included in the downloadable CSV. Repeated-fold tests refit calibration on earlier chronological development periods with three-day boundary gaps. A trial requires a test Brier interval wholly below zero, preserved discrimination and improvement on both development folds.</p>','<p>Coastal results use administrative South Coast and West Coast regions plus Coast Mountains district as a geographic proxy. They do not prove a meteorological maritime regime effect. Forecaster labels, operational low/moderate/high category fitting, subhour strike timing and multi-season validation remain deferred.</p>','<p>Additional diagnostics and exact values are available below. Archive inputs and operational forecasts were not modified by the study.</p>','<ul>'+''.join(f'<li><a href="{f}">{f}</a></li>' for f in ('manifest.json','selection.json','sample_inventory.csv','observation_inventory.csv','metrics.csv','bootstrap_differences.csv','rolling_development.csv','solar_models.json','solar_residuals.csv','ingredient_diagnostics.csv','first_event_offsets.csv','spatial_detail.csv','decision_gates.csv'))+'</ul></main></body></html>']
    (st.out/'report.html').write_text('\n'.join(parts))
    st.man['analysis_script_sha256']=digest(Path(__file__))
    st.man['analysis_completed_utc']=iso(dt.datetime.now(UTC))
    st.man['gaussian_support_rule']='4-sigma truncated kernel; normalized finite weights >= 0.95; finite center; 80 km grid-edge margin; common candidate/cross-horizon eligibility'
    write_json(st.out/'manifest.json',st.man)
    log('HTML report saved')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive-root',type=Path,default=archive.DEFAULT_ARCHIVE_ROOT); p.add_argument('--output',type=Path,default=Path('output/lpi_calibration/review_20261005')); p.add_argument('--phase',choices=['prepare','analyze','all'],default='all'); args=p.parse_args()
    if args.phase in ('prepare','all'): prepare(args.archive_root,args.output)
    if args.phase in ('analyze','all'):
        st=Study(args.output); run_core(st); run_neighborhood(st); run_solar(st); run_ingredients(st,args.archive_root); finalize_statistics(st); plots_and_cases(st); report(st)
    return 0


if __name__=='__main__': raise SystemExit(main())
