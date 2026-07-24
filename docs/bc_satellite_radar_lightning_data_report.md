# BC satellite, radar, and lightning imagery: feed and storage assessment

**Assessment date:** 2026-07-20
**Target:** BC at high quality, plus lower-resolution North America/Pacific overview imagery

## Executive recommendation

The proposed archive is practical. For display imagery, the limiting issue is **radar/lightning coverage**, not disk space.

- Use **GOES-18/West** for BC visible and IR; use GOES-19/East as well for an all-North-America mosaic. Add **Himawari-9** if “Pacific” means the whole basin rather than the eastern/central Pacific.
- Use ECCC’s **1-km North American radar composite**, **1-km surface precipitation type**, dynamic coverage mask, and four BC site images. Free Canadian “base” data are rendered CAPPI/DPQPE images, not true base reflectivity or velocity.
- Use ECCC’s **2.5-km, 10-minute CLDN flash-density** product for BC. GOES GLM is useful over oceans and southern BC, but its northern edge is roughly 52–54°N; it is not a BC-wide replacement.
- Store cropped/rendered assets, not full-disk/native inputs. The requested two-tier archive is about **1.5–2 GB** as quality JPEG/WebP plus transparent overlays. Provision **5 GB for the published archive** and **10 GB of local working space**. If every frame is a complete lossless PNG, provision **10–15 GB** instead.
- Treat all public feeds as operational but **without a public SLA**. Ingest continuously, retain valid-time metadata, show stale/missing states, and backfill from dated archives.

## Retention policy and frame counts

Assuming the week includes the most recent 24 hours and no duplicate boundary frame:

| Domain | Recent tier | Older tier | Retained slots/product |
|---|---:|---:|---:|
| BC | 24 h × 6 frames/h = 144 | preceding 6 d × 48 frames/d = 288 | **432** |
| North America/Pacific | 24 h × 2 frames/h = 48 | preceding 6 d × 24 frames/d = 144 | **192** |

Radar arrives every six minutes, so a common 10-minute animation should select the newest radar valid time not later than each animation time and preserve the actual source timestamp. Do not relabel or duplicate a previous image to hide a missing frame.

## Feed assessment

### Satellite: visible and IR

| Feed | Cadence and practical lag | Online archive | Quality and use |
|---|---|---|---|
| **NOAA GOES-18/19 ABI full disk** | 10 min. A full scan itself takes almost 10 min; NODD normally exposes the file about 10–11 min after its nominal scan-start time. | Full period of record in NOAA cloud/NCEI | Quantitative NetCDF radiance/reflectance/brightness temperature; fully restylable. ABI has 16 bands: 0.5-km red visible, 1-km principal visible/NIR, and 2-km thermal IR at nadir. [ABI overview](https://goes-r.noaa.gov/spacesegment/abi.html) · [band table](https://goes-r.noaa.gov/spacesegment/ABI-tech-summary.html) · [cloud archive](https://registry.opendata.aws/noaa-goes/) |
| **ECCC GOES GeoTIFF** | 10 min. On 2026-07-20, Natural Color was 13–22 min behind nominal time (median 14); Night IR was 12–20 min behind (median 17). | **30 days** through dated Datamart paths | Display-ready, georeferenced, internally JPEG-compressed RGB. Natural Color is nominally 1 km; Night IR 2 km. Excellent for plots, but not quantitative or freely restylable. [Datamart product guide](https://eccc-msc.github.io/open-data/msc-data/obs_satellite/readme_satellite-datamart_en/) |
| **ECCC GeoMet WMS/WCS** | Same 10-min processed products and similar lag | About **54 h** in the live capabilities during this audit | Best way to request one cropped BC/broad-domain image without downloading a full disk. [GeoMet layers](https://eccc-msc.github.io/open-data/msc-data/obs_satellite/readme_satellite_geomet_en/) |

For BC, GOES-18 at 137°W has excellent longitude geometry. Latitude still stretches the fixed grid:

| Product grid | Southern BC approximate ground sample | Northern BC approximate ground sample |
|---|---:|---:|
| 0.5-km red visible | 0.55 × 1.0 km | 0.55 × 1.4–1.5 km |
| 1-km natural colour | 1.1 × 2.0 km | 1.1 × 2.7–3.0 km |
| 2-km thermal IR | 2.2 × 4.0 km | 2.2 × 5.4–6.0 km |

Visible products are daylight-only. ECCC GOES-West visible products normally have no files around 06–11 UTC; that is expected, not an outage. IR is continuous. ECCC Natural Color is effectively limited by its 1-km colour components despite the 0.5-km red band.

High clouds require a warning when satellite is overlaid with radar or lightning. Across BC, a 10-km cloud top can be displaced roughly 16–25 km and a 15-km convective top roughly 24–37 km because of parallax. Native fixed-grid data are correctly navigated to the surface; the apparent cloud-top displacement is physical viewing geometry.

For the larger domain:

- GOES-18 is strong over western North America and the eastern/central Pacific.
- Blend GOES-18 and GOES-19 for consistently good North American resolution.
- Add Himawari-9 near the dateline for a whole-Pacific product; GOES-18 becomes strongly stretched toward the northwest Pacific. Himawari supplies 16 bands every 10 minutes at nominal 0.5–2 km. [JMA specifications](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html) · [public archive](https://registry.opendata.aws/noaa-himawari/)

### Radar: composite, site/base, and precipitation type

| Product | Cadence and lag | Online archive | Quality and limitation |
|---|---|---|---|
| **ECCC rain/snow composite** (`RADAR_1KM_RRAI`, `RADAR_1KM_RSNO`) | 6 min. Live valid-to-access lag was roughly 1–7 min; newest-image age therefore cycles upward between updates. | GeoMet: **3 h only**; there is no free bulk archive of this numeric composite. | True 1-km North American source grid, combining Canadian dual-pol data with U.S. MRMS. Keep the dynamic coverage mask so no-echo and no-coverage remain distinct. [GeoMet radar guide](https://eccc-msc.github.io/open-data/msc-data/obs_radar/readme_radar_geomet_en/) |
| **Surface precipitation type** (`Radar_1km_SfcPrecipType`) | 6 min; approximately **7 min production delay**, giving an effective newest-image age of about 7–13 min. | GeoMet: **3 h** | Rain, snow, freezing rain, hail, and mixed precipitation with intensity classes. It is model-assisted and uses a different overlap method from precipitation rate, so pixel-by-pixel comparison is discouraged. [Technical description](https://www.canada.ca/en/environment-climate-change/services/weather-general-tools-resources/radar-overview/about-radar.html) |
| **Free BC site imagery** | 6 min; Aldergrove files were posted in the same or following minute during the audit. | **30 d** Datamart | 580×480 rendered GIFs: low-sweep DPQPE precipitation rate and 1.0/1.5-km CAPPI. Useful as a local “base-like” view, but neither is true base reflectivity and neither includes velocity. [Site-image guide](https://eccc-msc.github.io/open-data/msc-data/obs_radar/readme_radarimage-datamart_en/) |
| **True Canadian base/volume data** | 6-min, 17-sweep volumes | No free online raw archive | ODIM-H5 reflectivity, radial velocity, and dual-pol moments are cost-recovered. Indicative 1–5-radar raw service: C$1,600/month plus C$2,000 setup. [Cost-recovered services](https://eccc-msc.github.io/open-data/cost-recovered/readme_en/) |
| **NOAA MRMS / NEXRAD** | MRMS about 2 min; NEXRAD volume commonly 4–6 min in precipitation mode | MRMS cloud archive from 2020; NEXRAD long-term archive | MRMS supplies digital 1-km-class `MergedBaseReflectivityQC`, composite reflectivity, precipitation rate, and precipitation-flag grids over the U.S. and southern Canada. KATX/KLGX/KOTX provide free true base reflectivity/velocity near the border, but cannot replace the Canadian network over BC. [MRMS](https://www.nssl.noaa.gov/projects/mrms/MRMS_data.php) · [NEXRAD](https://registry.opendata.aws/noaa-nexrad/) |

BC’s four sites are Aldergrove, Halfmoon Peak, Silver Star Mountain, and Prince George. Processed public range is 240 km; raw conventional range reaches 330 km. Coverage is strongest in southwest BC, the southern Interior, and near Prince George. It is weakest over northwest/far-northern BC, the north coast, mountain valleys, and terrain-shadowed sectors. At long range the beam can overshoot shallow valley precipitation. [Official site list](https://collaboration.cmc.ec.gc.ca/cmc/cmos/public_doc/msc-data/obs_radar/radars_list.pdf)

MRMS `PrecipFlag` is a QPE-regime classification (for example stratiform, convective, hail, or snow), not an equivalent backup for ECCC's freezing-rain/mixed-precipitation SPTP categories.

There is no meaningful radar coverage of the open Pacific. A Pacific overview should render radar only where a coverage mask says it exists, leaving the ocean blank rather than implying zero precipitation.

The three-hour GeoMet window is the most fragile archive dependency in this design. The desired week of composite and precipitation-type frames must be captured locally as they arrive. ECCC has an interactive historical rendered-image site, but not a free production bulk feed for reconstructing missed composite/SPTP frames; MRMS can partially reconstruct the southern domain. The 30-day Datamart retention applies to the free **site GIFs**, not the GeoMet composite.

### Lightning

| Feed | Cadence and lag | Online archive | Quality and limitation |
|---|---|---|---|
| **ECCC open CLDN density** | One 10-min accumulation every 10 min. Files were consistently posted 10 min after nominal time; depending on the timestamp convention, displayed flashes are roughly 10–20 min old. | GeoMet: **3 h**; dated Datamart: **30 d** | Best free BC-wide source. Underlying detections are located to a few hundred metres, but the public raster is a 2.5-km density grid with no strike polarity/current or individual strike records. Masked more than 250 km from Canadian land/sea boundaries. [Product definition](https://eccc-msc.github.io/open-data/msc-data/lightning/readme_lightning_en/) · [GeoTIFF access](https://eccc-msc.github.io/open-data/msc-data/lightning/readme_lightning-datamart_en/) |
| **GOES-18/19 GLM** | 20-s L2 files, required product latency under 20 s; aggregate locally to 10/30/60 min | Full NOAA archive from 2017 | Total lightning and excellent oceanic/storm-evolution context. Native footprint is about 8 km at nadir to 14 km near the field edge; a 2-km FED grid is oversampled, not 2-km information. Coverage ends around 52–54°N, so southern BC is marginal and northern BC is absent. [GLM characteristics](https://goes-r.noaa.gov/spacesegment/glm.html) |
| **Commercial ground networks** | Typically seconds to tens of seconds | Contract dependent | Paid CLDN/NALDN is the precise BC option. Vaisala GLD360 or Earth Networks ENTLN is needed for independent, basin-wide Pacific coverage. Confirm map-redistribution rights and SLA in the licence. [ECCC access FAQ](https://eccc-msc.github.io/open-data/faq/readme_en/#are-lightning-data-available) · [GLD360](https://www.vaisala.com/en/products/systems/lightning/gld360) |

CLDN density and GLM flash-extent density observe different phenomena and must not share a numerical colour scale. GLM is cloud-top optical total lightning, not a ground-strike locator.

## Disk-space and ingest estimates

### Finished display archive — recommended

Assumptions:

- BC context approximately 145–108°W, 45–63°N, rendered near 2048×996.
- Broad overview approximately 180–50°W, 5–75°N, rendered near 2400×1300.
- Satellite stored as quality JPEG/WebP; radar/lightning as transparent PNG; one static basemap stored separately.
- “Base” is one four-site BC montage, not four full standalone figures.

Measured on 2026-07-20, the BC WMS files were 0.47 MB visible, 0.26 MB IR, 69 KB radar, 16 KB precipitation type, and 8 KB lightning. The following budgets deliberately allow stormier scenes and higher JPEG/WebP quality:

| Frame-set component | BC budget/slot | Broad budget/slot |
|---|---:|---:|
| Visible background/mosaic | 1.00 MB | 1.50 MB |
| IR background/mosaic | 0.35 MB | 0.90 MB |
| Radar composite | 0.25 MB | 0.35 MB |
| BC site/base montage | 0.30 MB | — |
| Precipitation type | 0.10 MB | 0.15 MB |
| Lightning overlay | 0.05 MB | 0.10 MB |
| **Conservative total** | **2.05 MB** | **3.00 MB** |

| Archive | Calculation | Steady state |
|---|---:|---:|
| BC | 432 × 2.05 MB | **0.89 GB** |
| North America/eastern-central Pacific | 192 × 3.00 MB | **0.58 GB** |
| Whole-Pacific Himawari addition | about 192 × 1–2 MB | **0.2–0.4 GB** |
| **Recommended compressed total** | including small metadata/index overhead | **about 1.5–2 GB** |

New retained output is about **0.4–0.6 GB/day** before expiry. Allowing thumbnails, JSON manifests, atomic generations, missing-frame backfill, and temporary downloads, allocate **5 GB to published assets** and **10 GB working space**. If every product is delivered as a complete lossless PNG with coastlines and labels baked in, allocate **10–15 GB** instead.

### Source-data choices that change the answer

| Choice | Approximate impact |
|---|---:|
| Retain ECCC full-disk Natural Color (~27 MB) and Night IR (~5 MB) instead of crops | roughly **7–11 GB** for the BC 432-slot policy; approximately **3.5–4.6 GB/day** download at native 10-min cadence |
| Retain selected quantitative NOAA ABI bands for custom true colour + IR | roughly **100–250 GB per GOES** for the BC tier, depending on product/compression; tens of GB/day ingress |
| Retain all 16 native ABI channels | several hundred GB per GOES for the tier; not justified for display-only plots |
| Retain MRMS digital base/composite/rate/type sources at the broad tier | budget **1–3 GB** |
| Retain selected Level-II volumes from three U.S. border radars | budget **8–25 GB** |
| Retain all four paid Canadian raw volumes | about **10–16 GB/day**, or **70–112 GB/week** |
| Ingest GOES-18 + GOES-19 GLM continuously | **2–4 GB/day** transient ingress; **14–28 GB/week** only if raw 20-s files are unnecessarily retained |

For GLM, rendering every 30 minutes does not reduce raw ingress if every flash in the interval is required. Ingest each 20-s file, spatially filter/accumulate it, and delete it. The persisted accumulated grid is small.

## Reliability and backup design

| Feed family | Reliability assessment | Backup strategy |
|---|---|---|
| **Satellite** | Operational spacecraft/ground system, but spacecraft and processing outages occur. GOES-19 lost all products for 2 d 2 h in July 2026. Public cloud endpoints have no user SLA. | Primary NOAA NODD cloud plus a second cloud/NCEI for backfill; ECCC is a separately processed fallback. GOES-16 is NOAA’s on-orbit backup for GOES-18/19. Himawari-8 backs Himawari-9, but failover is not instantaneous. [Current GOES status](https://www.ospo.noaa.gov/operations/goes/status.html) · [July 2026 outage](https://www.ospo.noaa.gov/data/messages/2026/07/MSG_20260717_2344.html) |
| **Radar** | ECCC radar/Datamart/GeoMet are operational, but support is best-effort. Sites receive routine maintenance and have occasional power, telecom, mechanical, and technical outages. BC overlap is sparse. | Use ECCC contingency/coverage fields; MRMS and U.S. NEXRAD are partial backups near the border. No independent feed can recreate a failed Canadian radar over central/northern BC. [Outage policy](https://www.canada.ca/en/environment-climate-change/services/weather-general-tools-resources/radar-overview/outages-maintenance.html) |
| **Lightning** | CLDN runs continuously, but the free ECCC product has no public SLA. GLM provides independent physics only over southern BC and lower latitudes. | GeoMet and Datamart are alternate delivery paths, not independent observations. For true BC/Pacific redundancy, license GLD360 or ENTLN/CLDN point data. |

ECCC Datamart and GeoMet are documented as operational 24/7 with best-effort support. Dated Datamart paths currently retain 30 days, subject to available disk rather than an archive SLA. Use **AMQPS notification plus HTTPS fetch**, then dated paths for gap filling; do not depend on the short GeoMet time dimension. The `hpfx` mirror can be a secondary URL, but ECCC states that its Internet path lacks 24/7 redundancy. [Datamart service and retention](https://eccc-msc.github.io/open-data/msc-datamart/readme_en/)

Multiple NOAA clouds protect against a cloud access failure, not a spacecraft or upstream NOAA-production failure. Likewise, ECCC composite, MRMS, and station products can share the same underlying radar outage.

A same-day completeness snapshot illustrates why gap handling matters, but is not an uptime study. Through the audit cutoff, GOES-West Night IR had 133 of 135 expected timestamps; Natural Color had the same two unexplained misses plus its normal nighttime gap. Aldergrove DPQPE had 227/227 timestamps, CAPPI 226/227, and ECCC lightning 135/135. Public operational feeds are good, not perfect.

## Suggested implementation

1. **Ingest and backfill:** subscribe to ECCC AMQPS for satellite GeoTIFF, site radar GIF, and lightning GeoTIFF announcements. Query the GeoMet composite/SPTP separately and complete retries inside its three-hour window. Use NOAA NODD for native satellite/GLM and independent archive recovery.
2. **Render once:** request/crop one whole domain per product and time. Avoid WMS tile sweeps; ECCC's usage policy favours Datamart for archive building, so a sustained composite archive should use one cached domain request per layer/time and be confirmed with ECCC. Keep satellite as JPEG/WebP and observational overlays as transparent palette PNG. [Usage policy](https://eccc-msc.github.io/open-data/usage-policy/readme_en/)
3. **Compact deterministically:** retain all 10-min BC slots for 24 h, then only `:00`/`:30` through day 7. For the broad domain retain `:00`/`:30` for 24 h, then only `:00`.
4. **Preserve provenance:** sidecar JSON should include requested time, actual valid time, fetch time, source, source age, coverage state, checksum, and whether the frame is primary or fallback.
5. **Monitor content, not HTTP:** alert on source age, missing timestamps, unchanged hashes, radar coverage loss, and partial product generation. Never silently reuse a stale image under a new timestamp.

This design gives a fast week-long animation archive with a small disk footprint. ECCC's 30-day Datamart and NOAA's long-term archives can reconstruct satellite, lightning, and site/base inputs; the GeoMet radar composite/SPTP still requires dependable local capture.
