# Fire Activity Overlay

The two-panel fire-weather products display current fire activity on the
danger/lightning panel as a separate transparent web layer. Forecast PNGs do
not embed fire observations, so one lightweight overlay can be refreshed and
reused across every forecast hour.

## Source Priority

1. Use the BC Wildfire Service `Fire Locations - Current` ArcGIS layer and
   exclude incidents whose `FIRE_STATUS` is `Out`.
2. Add current U.S. wildfires from NIFC WFIGS/IRWIN. A NIFC failure does not
   suppress an otherwise valid BCWS overlay.
3. If the BCWS service is unavailable, use the NRCan/CWFIS
   `hotspots_last24hrs` WFS layer and retain detections assigned to BC.
4. If both live services fail, use a cached observation set no more than 12
   hours old. Otherwise render the model graphic without the overlay.

Agency-reported incidents use the same Lucide flame as the radar-satellite
display, with a thin black outline. BCWS incidents are red for Out of Control,
yellow for Being Held, and green for Under Control. Incidents without a
comparable status, including NIFC fires, are orange. Official BC Fires of Note
and current U.S. ICS-209 large incidents use a larger flame with a yellow halo.
The satellite fallback is aggregated into roughly 8-10 km cells and shown as
orange squares; this prevents repeat detections over one fire from obscuring
the forecast fields.

The active-fire and hotspot caches are considered fresh for 45 minutes. The
hourly job publishes four 1440x900 transparent overlays plus
`manifests/fire_activity.json`. PNGs are uploaded only when their pixels
change; the manifest is updated on every successful job. The viewer polls the
manifest every hour and applies the layer only to `latest` runs. In the
browser, the current base frame and live fire layer are flattened into a
cached PNG. The visible `<img>` therefore retains the live icons when it is
opened or copied on its own, without republishing every forecast hour whenever
fire activity changes.

A retrieval or cache-write failure is logged. A cached layer can be used for
up to 12 hours; after that the manifest marks the layer unavailable and the
viewer hides it.

## Completed Enhancements

- [x] Bring the fire-weather forecast overlays in line with the radar-satellite
  displays: colour-code fires by incident status, give Fires of Note a distinct
  emphasis, and add current U.S. fires with a compatible status mapping and
  source attribution.

Sources:

- BCWS: <https://services6.arcgis.com/ubm4tcTYICKBpist/ArcGIS/rest/services/BCWS_ActiveFires_PublicView/FeatureServer/0>
- NIFC WFIGS/IRWIN: <https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations_Current/FeatureServer/0>
- CWFIS: <https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows>
