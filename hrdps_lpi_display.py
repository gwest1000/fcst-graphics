"""Shared learned LPI field for the HRDPS convective four-panel display."""
from pathlib import Path
import hashlib
import numpy as np
import lpi_model
from project_paths import plot_path

def probability_grid(run_dir, run, fhour, lat, lon):
    # Lazy import: lightning diagnostics also import four-panel utilities.
    import make_hrdps_west_convective as h
    import make_hrdps_west_lightning as lightning
    key=h.model_config().key
    cache=plot_path(f'hrdps_{key}_lightning')/run.stamp/'lpi_cache'/f'hrdps_{key}_lightning_{run.stamp}_f{fhour:03d}_lpi.npz'
    expected=lpi_model.VERSION+('_initial' if fhour==0 else '_3h')
    model_sha=hashlib.sha256(lpi_model.MODEL_PATH.read_bytes()).hexdigest()
    if cache.exists():
        with np.load(cache) as z:
            required={'formula_version','run_stamp','fhour','model_key','model_sha256','lat','lon','potential'}
            if (required.issubset(z.files) and str(z['formula_version'].item())==expected and str(z['run_stamp'].item())==run.stamp
                and int(z['fhour'].item())==fhour and str(z['model_key'].item())==key
                and str(z['model_sha256'].item())==model_sha):
                return z['lat'].copy(),z['lon'].copy(),z['potential'].copy()
    # Four-panel rendering may run before Fire Weather. Compute on exactly the
    # same model subset as Fire Weather, not the four-panel's wider map extent.
    ys,xs=h.subset_slices(lat,lon,h.model_config().extent)
    terrain,_,_=h.read_grib(lightning.hour_file(Path(run_dir),run,h.TERRAIN_FHOUR,'HGT','SFC','0'))
    fields=lightning.compute_lightning_fields(Path(run_dir),run,fhour,ys,xs,lat,lon,terrain[ys,xs],7)
    stride=max(1,round(5/h.model_config().resolution_km))
    return lat[ys,xs][::stride,::stride],lon[ys,xs][::stride,::stride],fields.potential[::stride,::stride]
