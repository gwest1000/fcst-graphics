# Fire-danger products

This project deliberately separates two products that answer different questions.

## BC daily danger rating

The daily product uses the CWFIS FWI1987 FWI and BUI grids with the matrices in
Schedule 2 of the British Columbia Wildfire Regulation. Schedule 1 regions are
reconstructed as unions of maintained BC Geographic Warehouse Natural Resource
Districts corresponding to the former forest regions shown in the regulation:

- Region 2: 100 Mile House, Cariboo-Chilcotin, and Quesnel districts
- Region 3: Cascades, Okanagan Shuswap, Rocky Mountain, Selkirk, and Thompson Rivers districts
- Region 1: the remainder of British Columbia

This is the product closest to the familiar BC daily Danger Class. The current
district polygons are maintained official boundaries, but they are successors to
the former forest-district boundaries shown on the low-resolution Schedule 1 map.
The plot therefore describes this as a reconstruction rather than a legal GIS
boundary product.

## Experimental FWI2025 fire danger

The forecast product starts from the latest available CWFIS FFMC, DMC, and DC
grids. It advances all three moisture codes through every hourly HRDPS forecast
field using the NRCan FWI2025 equations, including cumulative storm interception,
five dry-hour rain-event reset, daylight-only DMC/DC drying, and the capped
high-wind ISI response. Three-hourly frames are selected from that hourly state.

The initial CWFIS codes are FWI1987 observations, so using them as an FWI2025
state reset is an approximation. A previous-run checkpoint or previous-run hourly
surface fields bridge the observed daily anchor to model initialization. The code
does not silently pretend that an older daily state is valid at initialization.

The shading uses NRCan's proposed national FWI2025 adjective breaks:

| Class | FWI2025 |
|---|---:|
| Low | 0 to less than 6 |
| Moderate | 6 to less than 16 |
| High | 16 to less than 23 |
| Very high | 23 to less than 30 |
| Extreme | 30 or greater |

This is not BC's regulatory Danger Class. In particular, it does not apply the
Schedule 2 FWI/BUI matrices, which were calibrated for FWI1987 and differ by BC
danger region. It is a forecast guidance layer for the timing and spatial pattern
of changing fire-weather conditions.

The plotter retains an optional `--classification schedule2` mode for research
comparisons. Frames produced in that mode are labelled as an experimental hourly
analog and must not be represented as an official BC danger rating.
