# Fire Activity Overlay

The two-panel fire-weather products display current fire activity on the
danger/lightning panel as a separate transparent web layer. Forecast PNGs do
not embed fire observations, so one lightweight overlay can be refreshed and
reused across every forecast hour.

## Source Priority

1. Use the BC Wildfire Service `Fire Locations - Current` ArcGIS layer.
2. Exclude incidents whose `FIRE_STATUS` is `Out`.
3. If the BCWS service is unavailable, use the NRCan/CWFIS
   `hotspots_last24hrs` WFS layer and retain detections assigned to BC.
4. If both live services fail, use a cached observation set no more than 12
   hours old. Otherwise render the model graphic without the overlay.

BCWS incidents use the same coral Lucide flame as the weather app, with a
thin black outline. Fires of Note use a larger flame. The satellite fallback
is aggregated into roughly 8-10 km cells and shown as orange squares; this
prevents repeat detections over one fire from obscuring the forecast fields.

The active-fire and hotspot caches are considered fresh for 45 minutes. The
hourly job publishes four 1440x900 transparent overlays plus
`manifests/fire_activity.json`. PNGs are uploaded only when their pixels
change; the manifest is updated on every successful job. The viewer polls the
manifest every five minutes and applies the layer only to `latest` runs.

A retrieval or cache-write failure is logged. A cached layer can be used for
up to 12 hours; after that the manifest marks the layer unavailable and the
viewer hides it.

Sources:

- BCWS: <https://delivery.maps.gov.bc.ca/arcgis/rest/services/mpcm/bcgwpub/MapServer/502>
- CWFIS: <https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows>
