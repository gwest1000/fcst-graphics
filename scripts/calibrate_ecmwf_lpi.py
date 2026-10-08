#!/usr/bin/env python3
"""Reproducible seven-day, nested whole-run transfer assessment and HTML report."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/fcstgraphics-mpl-cache')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union
import ecmwf_lpi as ec
import lpi_model as lpi
import lightning_ml_archive as archive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/ecmwf_lpi_transfer_20261007'
SITE = ROOT / 'output/lpi_calibration/review_20261005/site'
DATA = Path('/Volumes/Greg1_2tb/concrete_fcst_data')
STAMPS = [f'202610{day:02d}T12Z' for day in range(1,8)]
CANDIDATES = [('logit', r) for r in (.0001,.001,.01)] + [('feature_residual', r) for r in (.001,.01,.1)]


def teacher_cache():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {s.key:s for s in archive.HOURLY_LPI_INGREDIENT_SPECS}
    for stamp in STAMPS:
        target = OUT / f'teacher_{stamp}.npz'
        if target.exists():
            with np.load(target) as z:
                if str(z['teacher_sha256']) != hashlib.sha256(lpi.MODEL_PATH.read_bytes()).hexdigest():
                    raise ValueError('Teacher cache model changed')
            continue
        arrays = []
        sources = []
        for hour in range(1,25):
            path = archive.hourly_lpi_hour_archive_path(archive.DEFAULT_ARCHIVE_ROOT, stamp, hour)
            sources.append({'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            with np.load(path) as z:
                if str(z['grid_hash']) != lpi.load_model()['grid_hash']:
                    raise ValueError('Teacher ingredient grid mismatch')
                arrays.append(np.stack([archive.unpack_field(z[k], specs[k]) for k in lpi.KEYS]))
        a = np.stack(arrays)
        short = np.stack([lpi.infer(a[j:j+3], '3h')[0] for j in range(0,24,3)])
        daily, count = lpi.infer(a, '24h')
        ec.atomic_npz(target, short=short, daily=daily, count=count, teacher_sha256=np.asarray(hashlib.sha256(lpi.MODEL_PATH.read_bytes()).hexdigest()))
        (OUT / f'teacher_{stamp}.sources.json').write_text(json.dumps(sources,indent=2))
        print('TEACHER',stamp,flush=True)


def ingredient_coverage():
    target=OUT/'teacher_ingredient_coverage.json'
    if target.exists():
        return
    with np.load(archive.DEFAULT_ARCHIVE_ROOT/'baseline/hrdps_continental_lpi_5km/schema_v1/static/grid.npz') as z:
        lat,lon=z['lat'],z['lon']
    geo=json.loads((ROOT/'data/bc_danger_regions/natural_resource_districts.geojson').read_text())
    mask=contains_xy(unary_union([shape(f['geometry']) for f in geo['features']]),lon,lat)
    specs={s.key:s for s in archive.HOURLY_LPI_INGREDIENT_SPECS}
    result=[]
    for stamp in STAMPS:
        sums=np.zeros(10)
        for hour in range(1,25):
            with np.load(archive.hourly_lpi_hour_archive_path(archive.DEFAULT_ARCHIVE_ROOT,stamp,hour)) as z:
                sums += [np.isfinite(archive.unpack_field(z[k],specs[k]))[mask].mean() for k in lpi.KEYS]
        result.append({'stamp':stamp,'domain':'BC land points on archived HRDPS 5 km grid','finite_fraction_by_ingredient':dict(zip(lpi.KEYS,(sums/24).tolist()))})
    target.write_text(json.dumps(result,indent=2)+'\n')


def pairs():
    with np.load(archive.DEFAULT_ARCHIVE_ROOT / 'baseline/hrdps_continental_lpi_5km/schema_v1/static/grid.npz') as z:
        tla,tlo = z['lat'],z['lon']
    records = {'3h':[], '24h':[]}
    geo = json.loads((ROOT/'data/bc_danger_regions/natural_resource_districts.geojson').read_text())
    bc = unary_union([shape(f['geometry']) for f in geo['features']])
    for day,stamp in enumerate(STAMPS):
        with np.load(ec.archive_path(DATA,stamp)) as z:
            a,steps,lat,lon = z['inputs'],z['steps'],z['lat'],z['lon']
        # Each native ECMWF rectangular footprint receives an area mean of HRDPS
        # probabilities, not a nearest point or a fabricated independent 5-km sample.
        ys = lat[::-1,0]; xs = lon[0]
        yi = np.floor((tla - (ys[0]-.125))/.25).astype(int)
        xi = np.floor((tlo - (xs[0]-.125))/.25).astype(int)
        inside = (yi>=0)&(yi<len(ys))&(xi>=0)&(xi<len(xs))
        cell = ((len(ys)-1-yi)*len(xs)+xi)[inside]
        total = np.bincount(cell, minlength=lat.size).reshape(lat.shape)
        mask = contains_xy(bc, lon, lat) & (total >= 3)
        def aggregate(p):
            values = p[inside]; good = np.isfinite(values)
            n = np.bincount(cell[good], minlength=lat.size).reshape(lat.shape)
            s = np.bincount(cell[good], weights=values[good], minlength=lat.size).reshape(lat.shape)
            return np.divide(s,n,out=np.full(lat.shape,np.nan),where=(n>=3)&(n>=.8*total))
        with np.load(OUT/f'teacher_{stamp}.npz') as z:
            targets = {'3h':z['short'], '24h':z['daily'][None]}
        for duration in records:
            for block,target in enumerate(targets[duration]):
                hours = list(range(3,25,3)) if duration == '24h' else [(block+1)*3]
                ids = [int(np.flatnonzero(steps==s)[0]) for s in hours]
                f,count = ec.features(a[ids],lat)
                y = aggregate(target)
                good = mask & np.isfinite(y) & np.isfinite(f).all(0)
                record = {'day':day,'stamp':stamp,'end_hour':24 if duration=='24h' else (block+1)*3,'f':f,'y':y,'good':good,'weight':np.cos(np.deg2rad(lat)),'lat':lat,'lon':lon,'coverage':float(good.sum()/mask.sum()),'eligible_cells':int(mask.sum())}
                records[duration].append(record)
        print('PAIRED',stamp,'daily cells',records['24h'][-1]['good'].sum(),flush=True)
    return records


def samples(records, days, kind, duration):
    xs,ys,ws = [],[],[]
    for day in days:
        selected = [r for r in records if r['day']==day]
        denominator = sum(r['weight'][r['good']].sum() for r in selected)
        if denominator <= 0:
            raise ValueError(f'No matched data on day {day}')
        for r in selected:
            good = r['good'].ravel()
            xs.append(ec.design(r['f'],duration,kind)[good])
            ys.append(r['y'].ravel()[good]/100)
            ws.append(r['weight'].ravel()[good]/denominator/len(days))
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)


def fit(records,days,kind,ridge,duration):
    x,y,w = samples(records,days,kind,duration)
    prior = np.zeros(x.shape[1]); prior[1]=1
    def objective(beta):
        z=x@beta; p=expit(z)
        residual=beta-prior
        return np.sum(w*(np.logaddexp(0,z)-y*z))+.5*ridge*(residual@residual),x.T@(w*(p-y))+ridge*residual
    bounds = [(-12,12),(0,4)] + [(-8,8)]*(x.shape[1]-2)
    result = minimize(objective,prior,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def loss(records,day,kind,beta,duration):
    x,y,w = samples(records,[day],kind,duration)
    return float(np.sum(w*(expit(x@beta)-y)**2))


def select(records,days,duration):
    scores = []
    for kind,ridge in CANDIDATES:
        errors = [loss(records,d,kind,fit(records,[x for x in days if x!=d],kind,ridge,duration),duration) for d in days]
        scores.append({'kind':kind,'ridge':ridge,'mse':float(np.mean(errors))})
    return min(scores,key=lambda r:r['mse']),scores


def metrics(records,predictions):
    xs,ys,ws = [],[],[]
    usable = [d for d in range(7) if any(r['day']==d and r['good'].any() for r in records)]
    for day in usable:
        ids = [i for i,r in enumerate(records) if r['day']==day]
        denom = sum(records[i]['weight'][records[i]['good']].sum() for i in ids)
        for i in ids:
            r=records[i];good=r['good']
            xs.append(predictions[i][good]);ys.append(r['y'][good]);ws.append(r['weight'][good]/denom/len(usable))
    x,y,w = map(np.concatenate,(xs,ys,ws))
    mx,my = np.sum(w*x),np.sum(w*y)
    corr = np.sum(w*(x-mx)*(y-my))/np.sqrt(np.sum(w*(x-mx)**2)*np.sum(w*(y-my)**2))
    return {'rmse_pp':float(np.sqrt(np.sum(w*(x-y)**2))),'mae_pp':float(np.sum(w*np.abs(x-y))),'bias_pp':float(np.sum(w*(x-y))),'pearson':float(corr),'teacher_mean_pct':float(my),'prediction_mean_pct':float(mx),'pairs':len(x),'teacher_above20':int((y>=20).sum()),'teacher_max_pct':float(y.max())}


def evaluate(records):
    models,summary,details,preds = {},{},[],{}
    for duration,rows in records.items():
        usable = [d for d in range(7) if any(r['day']==d and r['good'].any() for r in rows)]
        before = [ec.predict(r['f'],duration,calibrated=False) for r in rows]
        after = [np.full(r['y'].shape,np.nan) for r in rows]; constant = [np.full(r['y'].shape,np.nan) for r in rows]
        for held in usable:
            train = [d for d in usable if d!=held]
            chosen,_ = select(rows,train,duration)
            beta = fit(rows,train,chosen['kind'],chosen['ridge'],duration)
            _,ty,tw=samples(rows,train,'logit',duration)
            for i,r in enumerate(rows):
                if r['day']==held:
                    after[i]=(100*expit(ec.design(r['f'],duration,chosen['kind'])@beta)).reshape(r['y'].shape)
                    constant[i]=np.full(r['y'].shape,100*np.sum(ty*tw))
            sub=[i for i,r in enumerate(rows) if r['day']==held]
            def daym(values):
                ww=np.concatenate([rows[i]['weight'][rows[i]['good']] for i in sub]);ww/=ww.sum()
                err=np.concatenate([(values[i]-rows[i]['y'])[rows[i]['good']] for i in sub])
                return {'rmse_pp':float(np.sqrt(np.sum(ww*err**2))),'mae_pp':float(np.sum(ww*np.abs(err))),'bias_pp':float(np.sum(ww*err))}
            details.append({'duration':duration,'stamp':STAMPS[held],'selection':chosen,'before':daym(before),'after':daym(after),'coverage':float(np.mean([rows[i]['coverage'] for i in sub]))})
            print('FOLD',duration,STAMPS[held],chosen,daym(before),daym(after),flush=True)
        chosen,candidates = select(rows,usable,duration)
        beta=fit(rows,usable,chosen['kind'],chosen['ridge'],duration)
        models[duration]={'kind':chosen['kind'],'ridge':chosen['ridge'],'beta':beta.tolist(),'selection_candidates':candidates}
        summary[duration]={'before':metrics(rows,before),'after_nested_holdout':metrics(rows,after),'constant_holdout':metrics(rows,constant),'mean_coverage':float(np.mean([r['coverage'] for r in rows])),'usable_stamps':[STAMPS[d] for d in usable],'excluded_stamps':[STAMPS[d] for d in range(7) if d not in usable]}
        preds[duration]={'before':before,'after':after}
    model={'version':ec.VERSION,'teacher_version':lpi.VERSION,'teacher_sha256':hashlib.sha256(lpi.MODEL_PATH.read_bytes()).hexdigest(),'models':models,'training_stamps_by_duration':{k:v['usable_stamps'] for k,v in summary.items()},'target':'Native ECMWF footprint mean of HRDPS probability; not independently verified ECMWF lightning probability','source_levels_hpa':list(ec.LEVELS),'feature_smoothing_sigma_km':20,'experimental':True,'transfer_accepted':False,'recommended_mode':'unadjusted_hrdps_recipe','calibrated_leads_by_duration':{'3h':list(range(3,25,3)),'24h':[24]},'short_window_after144':'Unsupported three-hour probability after F144; use complete daily products'}
    ec.MODEL_PATH.write_text(json.dumps(model,indent=2)+'\n')
    result={'summary':summary,'folds':details,'method':'Nested leave-one-issued-12Z-run-out; inner leave-one-run-out chooses formulation and regularization; day and area weighted','model_path':str(ec.MODEL_PATH),'model_sha256':hashlib.sha256(ec.MODEL_PATH.read_bytes()).hexdigest(),'period':'2026-10-01 12Z through 2026-10-08 12Z valid windows','issue_days':STAMPS,'limitations':['Seven runs archived; October 3 has no adequately covered teacher footprints, leaving six usable autumn weather days; spatial cells are correlated','Teacher agreement is not observed lightning skill','Transfer fit only assessed for lead hours 3–24 and 12Z runs','Daily and three-hour ECMWF summaries use eight and one native snapshots respectively; no synthetic hourly values','Most unstable LI is a sparse-profile parcel approximation; vertical and temporal resolution differ from HRDPS','Reported scores only include BC land cells with at least 80% finite teacher footprint coverage and finite ECMWF features']}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    report(result,records,preds)


def report(result,records,preds):
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=SITE/'ecmwf_lpi';dest.mkdir(parents=True,exist_ok=True)
    day=max(range(7),key=lambda d:float(np.nanmax(records['24h'][d]['y'])))
    r=records['24h'][day]
    fig,axes=plt.subplots(1,3,figsize=(13,5),constrained_layout=True)
    for ax,p,title in zip(axes,[r['y'],preds['24h']['before'][day],preds['24h']['after'][day]],['HRDPS teacher (cell mean)','ECMWF before transfer','ECMWF after transfer (day withheld)']):
        p=np.where(r['good'],p,np.nan)
        image=ax.pcolormesh(r['lon'],r['lat'],p,vmin=0,vmax=max(20,float(np.nanmax(r['y']))),cmap='magma',shading='auto')
        ax.set_title(title,fontsize=10);ax.set_xlabel('Longitude');ax.set_ylim(48,60);ax.set_xlim(-139,-114)
    axes[0].set_ylabel('Latitude');fig.colorbar(image,ax=axes,label='LPI (%)',shrink=.7)
    fig.savefig(dest/'comparison.png',dpi=150);plt.close(fig)
    s=result['summary'];d=s['24h'];b=d['before'];a=d['after_nested_holdout']
    change=100*(a['rmse_pp']/b['rmse_pp']-1)
    direction='increase' if change>=0 else 'reduction'
    def row(duration):
        z=s[duration];b=z['before'];a=z['after_nested_holdout'];c=z['constant_holdout']
        return f"<tr><td>{duration}</td><td>{b['rmse_pp']:.2f} → {a['rmse_pp']:.2f}</td><td>{b['mae_pp']:.2f} → {a['mae_pp']:.2f}</td><td>{b['bias_pp']:+.2f} → {a['bias_pp']:+.2f}</td><td>{b['pearson']:.3f} → {a['pearson']:.3f}</td><td>{c['rmse_pp']:.2f}</td><td>{z['mean_coverage']:.1%}</td></tr>"
    folds=''.join(f"<tr><td>{r['stamp']}</td><td>{r['before']['rmse_pp']:.2f}</td><td>{r['after']['rmse_pp']:.2f}</td><td>{r['coverage']:.1%}</td><td>{r['selection']['kind']}; ridge {r['selection']['ridge']}</td></tr>" for r in result['folds'] if r['duration']=='24h')
    limitations=''.join('<li>'+html.escape(x)+'</li>' for x in result['limitations'])
    model=json.loads(ec.MODEL_PATH.read_text())
    candidate_rows=''.join(f"<tr><td>{k}</td><td>{r['kind']}</td><td>{r['ridge']}</td><td>{100*np.sqrt(r['mse']):.2f}</td></tr>" for k,v in model['models'].items() for r in v['selection_candidates'])
    coverage_rows=json.loads((OUT/'teacher_ingredient_coverage.json').read_text())
    coverage_table=''.join(f"<tr><td>{r['stamp']}</td><td>{r['finite_fraction_by_ingredient']['cape']:.1%}</td></tr>" for r in coverage_rows)
    fittext='; '.join(f"{k}: {v['kind']}, regularization {v['ridge']}" for k,v in model['models'].items())
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECMWF LPI transfer calibration</title><style>body{{font:17px/1.65 system-ui;max-width:1100px;margin:35px auto;padding:0 22px;color:#183042;background:#f4f7fa}}h1,h2{{line-height:1.2}}section{{background:white;border-radius:12px;padding:24px;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}img{{width:100%}}.big{{font-size:25px;font-weight:650}}a{{color:#1264a3}}</style>
    <h1>ECMWF LPI: matching the HRDPS formulation</h1><p>Seven 12Z runs issued 1–7 October 2026. Valid daily windows extend from 1 October 12Z to 8 October 12Z. Experimental transfer model.</p>
    <section><h2>Headline results</h2><p class="big">Daily disagreement: {b['rmse_pp']:.2f} → {a['rmse_pp']:.2f} percentage points ({abs(change):.1f}% {direction} in RMSE).</p><p>This compares ECMWF with HRDPS on days excluded from fitting. It measures how closely the two model products agree. It does not demonstrate an improvement in forecasts of actual lightning.</p><p>Daily correlation changed from {b['pearson']:.3f} to {a['pearson']:.3f}. A constant prediction trained on the other days has RMSE {d['constant_holdout']['rmse_pp']:.2f} percentage points. This baseline matters: merely reducing probabilities in a quiet week can make average errors smaller without locating storms better.</p><p>The HRDPS teacher averaged {b['teacher_mean_pct']:.2f}% and reached {b['teacher_max_pct']:.1f}% across scored daily cells. {b['teacher_above20']:,} daily cell cases reached 20% or higher.</p></section>
    <section><h2>What goes into the ECMWF version?</h2><p>I retained the HRDPS recipe’s ten ingredients and twenty feature definitions. The main change is reconstructing them from the ECMWF fields that actually exist:</p><ol><li><b>Instability:</b> MUCAPE plus an approximate most-unstable lifted index. The lifted index selects the most buoyant parcel among the surface and available pressure levels within 300 hPa of the ground.</li><li><b>Charging layer:</b> temperature and relative humidity at nine source levels, 1000–250 hPa. These are interpolated in log pressure to the HRDPS layer quadrature. The same −20 to 0°C charging interval, −15°C thermal weighting, pressure-depth calculation, and −30 to −5°C mid-level humidity interval are used. Layers are clipped at the model surface pressure.</li><li><b>Ascent:</b> upward geometric velocity converted from pressure velocity at 500 and 700 hPa, using temperature and pressure.</li><li><b>Rain:</b> differences in accumulated precipitation and instantaneous precipitation rate. Accumulations are converted to a three-hour equivalent, with no negative rainfall.</li><li><b>Low-level moisture:</b> surface relative humidity from 2-m temperature/dewpoint and mean above-ground subcloud humidity at the surface, 850, 800, 750 and 700 hPa.</li><li><b>Space and time:</b> the same 20-km Gaussian feature smoothing, expressed in kilometres on ECMWF’s latitude-dependent grid. Daily summaries use eight three-hourly snapshots. A short-window summary uses its endpoint snapshot. Neither can reproduce all hourly peaks captured by HRDPS.</li></ol><p>Source: <a href="https://www.ecmwf.int/en/forecasts/datasets/open-data">ECMWF IFS Open Data</a>, operational control stream oper/fc, 0.25°. Missing profile levels were fetched from the source for all seven runs. Compact regional ingredient archives persist outside the raw-data cleanup directory.</p></section>
    <section><h2>What “before” and “after” mean</h2><p><b>Before:</b> feed the reconstructed ECMWF summaries through the frozen HRDPS probability model, without changing its weights.</p><p><b>After:</b> learn a conservative correction toward HRDPS. I compared a simple intercept/slope correction in log-odds with an ingredient-dependent residual correction using the same twenty features. Each had three regularization strengths, which limit how far its coefficients can move from the unmodified recipe. Final production fits: {fittext}.</p><p>To prevent the reported improvement from being a fitting illusion, each entire issued day was withheld. October 3 has almost no finite HRDPS probabilities and no usable matched footprints, so six days are usable. The other five usable days were used to choose the correction, with a second layer of whole-day withholding inside those five days. Then the correction was fitted on those five days and predicted the unseen sixth day. This was repeated six times. The final saved coefficients use all six usable days; their in-sample scores are not the reported scores.</p><p>Each ECMWF cell is compared with the mean HRDPS probability over its geographic footprint. At least 80% of its archived HRDPS points must have finite probabilities. Only BC land cells with finite inputs are scored. Cells are weighted by area and each issued day gets equal total weight. Spatial pixels are not treated as independent weather cases.</p></section>
    <section><h2>Agreement before and after</h2><div class="scroll"><table><tr><th>Window</th><th>RMSE (pp)</th><th>MAE (pp)</th><th>Bias (pp)</th><th>Correlation</th><th>Constant RMSE</th><th>Coverage</th></tr>{row('24h')}{row('3h')}</table></div><p><b>RMSE:</b> typical disagreement with extra weight on large misses. <b>MAE:</b> average absolute disagreement. <b>Bias:</b> ECMWF minus HRDPS, so a positive value means ECMWF is higher. “pp” means percentage points: 12% versus 10% differs by 2 points. <b>Correlation:</b> whether high and low values occur together, from −1 to +1. <b>Coverage:</b> the share of eligible BC cells usable by both methods; unscored cells remain missing.</p><p>There are {b["pairs"]:,} matched daily cell cases and {s["3h"]["before"]["pairs"]:,} short-window cell cases. These provide many spatial comparisons, but only six independent issued weather days. October 3 contributes no scored cells. Daily coverage averages {d["mean_coverage"]:.1%} across the seven archived days; this is a patchy sample of BC, not a complete provincial assessment.</p><h3>What causes the teacher coverage gaps?</h3><p>The main bottleneck is archived HRDPS CAPE. This table shows its finite coverage over BC grid points, averaged across the 24 forecast hours. A missing CAPE value prevents the existing HRDPS recipe from using that hour. I have not assumed that missing values mean zero instability; whether they originate in the source field or archive processing requires a separate check.</p><table><tr><th>Issued run</th><th>Finite hourly HRDPS CAPE coverage</th></tr>{coverage_table}</table><p>Hourly ingredient coverage differs from daily scored coverage: daily probabilities use available hours and must also pass spatial support checks. <a href="ecmwf_lpi/teacher_ingredient_coverage.json">Detailed ingredient coverage</a>.</p><h3>Every fitted correction tried</h3><div class="scroll"><table><tr><th>Window</th><th>Correction</th><th>Regularization</th><th>Fixed-method held-out RMSE (pp)</th></tr>{candidate_rows}</table></div><p>This table holds each correction and its regularization strength fixed, then withholds each usable day in turn. Every correction has worse daily RMSE than the unadjusted 3.28-point baseline. The main “after” score also includes the uncertainty of choosing the correction on the other days, which is why it differs from these fixed-method scores.</p><h3>Daily results, one withheld run at a time</h3><div class="scroll"><table><tr><th>Issued run</th><th>Before RMSE</th><th>After RMSE</th><th>Coverage</th><th>Correction selected without this day</th></tr>{folds}</table></div></section>
    <section><h2>A map comparison</h2><p>The daily window from {r['stamp']} is shown because it contains the highest teacher value in this sample. All panels use the same scale and common valid cells. The after-transfer panel was predicted while that entire day was excluded from fitting.</p><img src="ecmwf_lpi/comparison.png" alt="Matched daily HRDPS and ECMWF probability maps"></section>
    <section><h2>Why the correction failed</h2><p>On October 2, calibration increases daily RMSE from 1.85 to 6.77 points: adjustments learned from the other weather days do not transfer to that day. On October 5, the strongest teacher pattern remains largely absent from ECMWF, and daily error stays near 7.5 points. Raising the overall probability scale cannot recover a missing or displaced weather pattern. These examples explain why the improved average bias is not sufficient grounds to adopt the correction.</p><h2>Implementation status</h2><p>The runtime supports archiving all native ECMWF forecast steps through F360 and generating complete 24-hour prototype windows throughout that horizon. The standard ECMWF automation attempts to retain ingredients for future processed runs before raw-data cleanup. Public ECMWF images retain their current display. The fitted correction requires an explicit research flag; it is not the default.</p><h2>Recommendation and limits</h2><p>Do not apply the fitted transfer correction operationally. It reduces mean bias but increases held-out daily RMSE and loses spatial agreement. Keep the reconstructed ECMWF formulation as an experimental, unadjusted HRDPS-scale LPI; the saved correction remains available for explicit research use. Published ECMWF panels have not been switched on the strength of this failed calibration. Retain the compact ingredients and accumulate an ECMWF archive before refitting against observed lightning. Six usable autumn days do not establish summer performance, seasonal adjustments, or reliable day-two and longer-lead calibration.</p><p>The transfer is assessed only for the first 24 forecast hours of 12Z runs. Applying it at other cycles or longer leads is an extrapolation. After F144, ECMWF’s six-hour cadence cannot be labelled a calibrated three-hour lightning probability. The implementation therefore rejects three-hour products after F144 and supports complete 24-hour products from the native six-hour samples.</p><p>HRDPS teacher probabilities follow the production available-hour rule: daily values may summarize fewer than 24 finite ingredient hours. All 24 input files exist, but this does not guarantee complete temporal support at every location.</p><ul>{limitations}</ul><p><a href="ecmwf_lpi/results.json">Download detailed results</a> · <a href="ecmwf_lpi/model.json">Download fitted model</a> · <a href="report.html">HRDPS study</a></p></section></html>'''
    (SITE/'ecmwf_lpi_transfer.html').write_text(page)
    import shutil
    shutil.copy2(OUT/'teacher_ingredient_coverage.json',dest/'teacher_ingredient_coverage.json');shutil.copy2(OUT/'results.json',dest/'results.json');shutil.copy2(ec.MODEL_PATH,dest/'model.json')
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--teacher-only',action='store_true');args=ap.parse_args()
    teacher_cache()
    ingredient_coverage()
    if not args.teacher_only:
        evaluate(pairs())
