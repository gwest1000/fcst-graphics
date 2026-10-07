# BC LPI version 4

The implemented LPI is `bc_lpi_v4_random32`. It predicts the probability of at
least one lightning occurrence within 30 km, expressed as 0-100 percent.
Separate models predict the full day and each three-hour block. The current
activation covers continental HRDPS 12Z runs at F003 through F024. F000,
other cycles, day two and HRDPS-West are outside the evaluated deployment scope.

## Repository and runtime

This implementation lives in the existing repository:
https://github.com/gwest1000/fcst-graphics

`lpi_model.py` requires NumPy and SciPy for inference. The fitted model is a
plain JSON file at `models/lpi/bc_lpi_v4.json`; inference does not import the
research scripts or require a machine-learning framework. Existing graphics
packages are used only for rendering. Coefficients are read once per process.
A model change requires a new model version and restarting long-lived workers.

## Inputs and processing

The ten hourly physical inputs are most-unstable lifted index, CAPE,
charging-layer humidity, weighted charging-layer pressure depth, mid-level
humidity, resolved upward velocity, trailing three-hour rain, current rain
rate, surface humidity and subcloud humidity. Sampling and quantisation match
the archived 5 km training data exactly, including its BC-plus-buffer mask.

The twenty features comprise seven component peaks, three means, two fractions
of favourable hours, six joint ingredient peaks, and two humidity peaks. Their
names and order are frozen in the model specification. Each feature is
Gaussian-smoothed with sigma 20 km. A valid centre and at least 95% valid
kernel support are required. A forecast hour contributes only where all ten
source fields are finite. Missing hours do not count as zero. Available-hour
means and fractions are normalised by the valid-hour count.

The output applies a logistic conversion to a weighted combination of the
20 features and 32 fixed seeded tanh responses. Standardisation statistics
and the output weights are fitted; hidden weights are fixed. Training uses
log loss and L2 penalty 0.001. A separate fit is used for each duration.

The final coefficients use all 62 cases in the frozen archive through the
5 October 2026 cutoff. This is a final refit after the architecture was selected;
its coefficients are not the same as the independently fitted chronological
research models. The reported verification gains belong to those earlier
chronological tests, not an independent test of the final refit. The selected
architecture achieved 19.53% pooled daily Brier improvement against the matched
original recipe at 20 km smoothing, and 2.75% against the preceding learned
winner. The additional improvement did not pass every period's uncertainty gate.

## Operational outputs

The three-hour maps use a direct block probability, not the maximum hourly
index. Coarse probabilities are interpolated only for display on the native
grid. The BC two-panel plot labels contours as probabilities and does not
apply another spatial smoothing pass. Supported caches contain the formula
version, target radius, time representation and model SHA-256.

At F024 the worker reads all 24 retained hourly ingredient files, verifies their
checksums and grid identity, writes `*_f024_lpi24h.npz`, and renders a
`*_lightning_daily_*_f024.png` map. Daily verification consumes that direct
cache. It never takes the maximum of three-hour probabilities. If hourly files
are pending, the daily output is deferred; if the direct cache is absent,
daily verification skips that product. Grid mismatches fail explicitly.
The existing archive of hourly diagnostic ingredients retains its own version.

To create a daily forecast from an existing ingredient archive:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_lpi_probability.py \
  --stamp 20261007T12Z --output-dir output/lpi_v4_current
```

The map shows low probabilities and missing support separately. Some derived
humidity fields are undefined in clear or cold profiles despite complete
archive files; those locations remain missing. This is a substantial coverage
limitation, especially in autumn, and must not be interpreted as zero risk.

## Dry lightning

Dry lightning remains a screening score rather than a probability of dry
strikes or ignitions. For each hour, use the smaller of surface and subcloud
RH. Its dryness factor increases linearly from zero at 55% RH to one at 30% RH.
Use the greatest hourly dryness factor in the three-hour block. Where the
three-hour LPI probability is at least 20%, multiply that probability by the
dryness factor, then reduce the score linearly as block rain increases from
0.25 to 2.5 mm. Missing humidity or rain stays missing. Plot an asterisk when
the score reaches 15. On the BC two-panel map, use grey below LPI 60% and
black at or above 60%. These display thresholds have not been independently
recalibrated for dry-lightning observations.

## Verification and reproducibility

`tests/check_lpi_model_deployment.py` compares production feature calculations
against full-grid research results, checks probability parity, activation scope,
missing-data handling, and direct daily verification. Existing diagnostic,
verification and two-panel tests cover the graphics integration.

The research scripts use large frozen arrays in `output/lpi_calibration/`,
which are not stored in Git. `scripts/fit_lpi_operational_model.py` refits the
fixed architecture from those arrays. A forecaster reference is generated by
`scripts/build_lpi_v4_quick_reference.py`; its checked PDF is retained under
`docs/reference/`.
