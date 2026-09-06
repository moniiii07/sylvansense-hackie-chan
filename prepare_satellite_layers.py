import json
import ee
import geemap

ee.Initialize(project="composed-arch-476417-e5")

with open("target_forest.geojson") as file:
    region = ee.Geometry(json.load(file))


def mask_s2_clouds(image):
    scl = image.select("SCL")
    clear = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
    )
    return image.updateMask(clear)


# Optical imagery: Sentinel-2, June–September 2024
sentinel_2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate("2024-06-01", "2024-09-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    .map(mask_s2_clouds)
    .median()
    .clip(region)
)

ndvi = sentinel_2.normalizedDifference(["B8", "B4"]).rename("NDVI")

# Radar imagery: Sentinel-1 VV and VH polarization
sentinel_1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(region)
    .filterDate("2024-06-01", "2024-09-30")
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .select(["VV", "VH"])
    .median()
    .clip(region)
)

center = [-3.17613, -60.16615]
Map = geemap.Map(center=center, zoom=12)

Map.addLayer(
    sentinel_2,
    {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000},
    "Sentinel-2 true color",
)
Map.addLayer(
    ndvi,
    {"min": 0, "max": 0.9, "palette": ["brown", "yellow", "lightgreen", "darkgreen"]},
    "Sentinel-2 NDVI",
)
Map.addLayer(
    sentinel_1.select("VV"),
    {"min": -25, "max": 0},
    "Sentinel-1 VV radar",
)
Map.addLayer(
    sentinel_1.select("VH"),
    {"min": -30, "max": -5},
    "Sentinel-1 VH radar",
)
Map.addLayer(region, {"color": "cyan"}, "Target forest boundary")

Map.to_html("target_satellite_layers.html")
print("Created target_satellite_layers.html")