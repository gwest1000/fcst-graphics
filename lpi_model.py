"""Frozen BC LPI probability model: portable inference without research imports."""
from __future__ import annotations
import hashlib,json,os,warnings
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.special import expit

VERSION='bc_lpi_v4_random32'
MODEL_PATH=Path(__file__).resolve().parent/'models/lpi/bc_lpi_v4.json'
KEYS=('mu_li','cape','charge_rh','charge_depth','mid_rh','upward_w','precip_3h','precip_rate','surface_rh','subcloud_rh')
ATTRS=('li','cape','charge_rh','charge_depth_hpa','mid_rh','upward_w_ms','precip_3h','precip_rate_mm_h','surface_rh','subcloud_rh')
SCALES=(.01,1.,.01,.05,.01,.001,.01,.01,.01,.01)
FEATURES=('li_peak','cape_peak','charge_rh_peak','charge_depth_peak','mid_rh_peak','ascent_peak','rain_peak','li_mean','charge_mean','ascent_mean','unstable_fraction','ascent_fraction','instability_charge_peak','charged_ascent_peak','convective_rain_peak','cape_ascent_peak','dry_ascent_peak','humid_surface_peak','humid_subcloud_peak','li_ascent_peak')

def supported(model_key,cycle,fhour):
 return model_key=='continental' and cycle=='12' and 0<int(fhour)<=24 and int(fhour)%3==0

def ramp(a,lo,hi):return np.clip((a-lo)/(hi-lo),0,1)

@lru_cache(maxsize=1)
def load_model():
 model=json.loads(MODEL_PATH.read_text())
 if model['version']!=VERSION or model['features']!=list(FEATURES):raise ValueError('Incompatible LPI model specification')
 return model

def grid_hash(lat,lon):
 h=hashlib.sha256();h.update(np.asarray(lat,dtype=np.float32).tobytes());h.update(np.asarray(lon,dtype=np.float32).tobytes());return h.hexdigest()

def snapshot_inputs(snapshots,stride=2,domain_mask=None):
 """Match the 5 km training archive's sampling and int16 physical precision."""
 fields=[]
 for s in snapshots:
  aa=[]
  for attr,scale in zip(ATTRS,SCALES):
   value=getattr(s,attr)
   if value is None:raise ValueError(f'LPI requires {attr}')
   a=np.asarray(value[::stride,::stride],dtype=np.float32)
   a=np.clip(np.rint(a/scale),-32767,32767).astype(np.float32)*scale
   if domain_mask is not None:a=np.where(domain_mask,a,np.nan)
   aa.append(a)
  fields.append(np.stack(aa))
 return np.stack(fields)

def features(a):
 """20 temporal summaries, then Gaussian sigma20 km on the 5 km grid."""
 a=np.asarray(a,dtype=np.float32)
 if a.ndim!=4 or a.shape[1]!=10 or len(a) not in (3,24):raise ValueError('Expected 3 or 24 hourly arrays with ten ingredients')
 valid=np.isfinite(a).all(1);count=valid.sum(0)
 z=[ramp(a[:,0],1.,-5.),ramp(a[:,1],75,800),ramp(a[:,2],45,80),ramp(a[:,3],35,150),ramp(a[:,4],35,75),ramp(a[:,5],.005,.05),np.maximum(ramp(a[:,6],.05,1.5),ramp(a[:,7],.02,.8)),np.clip(a[:,8]/100,0,1),np.clip(a[:,9]/100,0,1)]
 li,cape,rh,dep,mid,u,rain,surface,sub=[np.where(valid,x,np.nan) for x in z];charge=np.sqrt(rh*dep)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  peak=lambda x:np.nanmax(x,axis=0)
  mean=lambda x:np.nanmean(x,axis=0)
  fraction=lambda x:np.divide(x.sum(0),count,out=np.full(count.shape,np.nan),where=count>0)
  f=np.stack([peak(li),peak(cape),peak(rh),peak(dep),peak(mid),peak(u),peak(rain),mean(li),mean(charge),mean(u),fraction(li>.25),fraction(u>.2),peak(li*charge),peak(li*charge*u),peak(li*charge*rain),peak(cape*charge*u),peak(li*charge*u*(1-sub)),peak(surface),peak(sub),peak(li*u)]).astype(np.float32)
 out=[]
 for x in f:
  good=np.isfinite(x);weight=gaussian_filter(good.astype(np.float32),4.,mode='constant',cval=0,truncate=4.)
  num=gaussian_filter(np.where(good,x,0),4.,mode='constant',cval=0,truncate=4.)
  y=np.divide(num,weight,out=np.full_like(x,np.nan),where=weight>0)
  out.append(np.where(good&(weight>=.95),y,np.nan))
 return np.stack(out),count

def probability(feature_grid,duration,model=None):
 model=load_model() if model is None else model
 if duration not in ('24h','3h'):raise ValueError('Duration must be 24h or 3h')
 m=model['models'][duration];X=np.asarray(feature_grid).reshape(20,-1).T;good=np.isfinite(X).all(1);result=np.full(len(X),np.nan,dtype=np.float32)
 W=np.asarray(m['W']);bias=np.asarray(m['bias']);beta=np.asarray(m['beta']);mu=np.asarray(m['mean']);sd=np.asarray(m['sd'])
 for ids in np.array_split(np.flatnonzero(good),max(1,int(np.ceil(good.sum()/5000)))):
  x=X[ids];hidden=np.tanh(((x-mu)/sd)@W+bias);result[ids]=100*expit(beta[0]+x@beta[1:21]+hidden@beta[21:])
 return result.reshape(feature_grid.shape[1:])

def infer(a,duration,model=None):
 expected=24 if duration=='24h' else 3
 if len(a)!=expected:raise ValueError(f'{duration} requires exactly {expected} forecast hours')
 f,count=features(a);return probability(f,duration,model),count

def native_grid(coarse,shape,stride=2):
 """Interpolate display only; retain missing support and exact sampled values."""
 row,col=np.indices(shape,dtype=np.float32);coords=np.array([row/stride,col/stride]);valid=np.isfinite(coarse)
 weight=map_coordinates(valid.astype(np.float32),coords,order=1,mode='nearest')
 values=map_coordinates(np.where(valid,coarse,0),coords,order=1,mode='nearest')
 return np.where(weight>=.999,values,np.nan).astype(np.float32)

def dry_score(potential,snapshots,precip_3h,gate=20.):
 """Heuristic dry-lightning screening score; not a calibrated probability."""
 factors=np.stack([ramp(55-np.minimum(s.surface_rh,s.subcloud_rh),0,25) for s in snapshots])
 valid=np.isfinite(factors).any(0);dry=np.max(np.where(np.isfinite(factors),factors,-np.inf),axis=0)
 score=np.where(potential>=gate,potential*dry,0)*(1-ramp(precip_3h,.25,2.5))
 return np.where(valid&np.isfinite(potential)&np.isfinite(precip_3h),np.clip(score,0,100),np.nan).astype(np.float32)

def write_cache(path,run,duration,lat,lon,potential,hour_counts):
 """Atomic, self-describing probability cache; duration is never implicit."""
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(f'.{os.getpid()}.tmp')
 fhour=24 if duration=='24h' else None
 if fhour is None:raise ValueError('This cache writer is for complete day-one products')
 try:
  with tmp.open('wb') as f:np.savez_compressed(f,version=np.asarray([4],np.int16),formula_version=np.asarray(VERSION+'_'+duration),model_key=np.asarray('continental'),model_label=np.asarray('HRDPS 2.5 km'),source_label=np.asarray('ECCC HRDPS'),run_stamp=np.asarray(run.stamp),init_iso=np.asarray(run.init_time.isoformat()),fhour=np.asarray([24],np.int16),window_fhours=np.arange(1,25,dtype=np.int16),temporal_aggregation=np.asarray('direct_daily_probability'),target_radius_km=np.asarray([30],np.int16),probability_scale=np.asarray('percent'),feature_smoothing_sigma_km=np.asarray([20],np.float32),model_sha256=np.asarray(hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()),lat=lat,lon=lon,potential=potential,valid_ingredient_hours=hour_counts)
  tmp.replace(path)
 finally:tmp.unlink(missing_ok=True)

def plot_daily(path,run,lat,lon,p):
 """Render the direct daily forecast; missing support is masked."""
 init=run.init_time
 import datetime as dt
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import cartopy.crs as ccrs
 import make_hrdps_west_lightning as lightning
 import make_hrdps_west_convective as hrdps
 hrdps.set_model('continental');lightning.set_model('continental');fig=plt.figure(figsize=(10,10));ax=fig.add_subplot(111,projection=lightning.PLOT_CRS);ax.set_extent((-138.2,-114.,48.,60.),crs=lightning.DATA_CRS);hrdps.add_map_features(ax)
 levels=[0,5,10,20,30,40,60,80,100];cs=ax.contourf(lon,lat,np.ma.masked_invalid(p),levels=levels,cmap='Purples',transform=lightning.DATA_CRS,transform_first=True);fig.colorbar(cs,ax=ax,shrink=.65,label='24-h probability of lightning within 30 km (%)');ax.set_title(f'BC LPI | 24-hour lightning probability\n{init:%d %b %Y %H}Z to {(init+dt.timedelta(days=1)):%d %b %Y %H}Z');fig.text(.08,.045,'Missing support is blank. 12Z continental HRDPS, day one. Gaussian feature sigma 20 km.',fontsize=9);png=Path(path);fig.savefig(png,dpi=150,bbox_inches='tight');plt.close(fig)
 return png
