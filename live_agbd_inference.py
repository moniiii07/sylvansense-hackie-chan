import json
import ee
import geemap

ee.Initialize(project="composed-arch-476417-e5")

with open("target_forest.geojson") as file:
    target_region = ee.Geometry(json.load(file))

training_region = target_region.centroid(maxError=1).buffer(100000)

feature_bands = [
    "B2", "B3", "B4", "B8", "B11", "B12",
    "NDVI", "VV", "VH", "VV_minus_VH",
]


def mask_s2_clouds(image):
    scl = image.select("SCL")
    clear = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
    )
    return image.updateMask(clear)


def mask_gedi_quality(image):
    return (
        image.updateMask(image.select("l4_quality_flag").eq(1))
        .updateMask(image.select("degrade_flag").eq(0))
        .updateMask(image.select("algorithm_run_flag").eq(1))
    )


s2_full = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(training_region)
    .filterDate("2024-06-01", "2024-09-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    .map(mask_s2_clouds)
    .median()
    .clip(training_region)
)

s2 = s2_full.select(["B2", "B3", "B4", "B8", "B11", "B12"])
ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")

s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(training_region)
    .filterDate("2024-06-01", "2024-09-30")
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .select(["VV", "VH"])
    .median()
    .clip(training_region)
)

vv_minus_vh = s1.select("VV").subtract(s1.select("VH")).rename("VV_minus_VH")
predictors = s2.addBands(ndvi).addBands(s1).addBands(vv_minus_vh)

agbd = (
    ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY")
    .filterBounds(training_region)
    .filterDate("2019-04-01", "2024-11-30")
    .map(mask_gedi_quality)
    .select("agbd")
    .mean()
    .rename("agbd")
    .clip(training_region)
)

valid_gedi = agbd.mask().rename("valid_gedi").unmask(0).toByte()

training_stack = (
    predictors
    .addBands(agbd.unmask(-9999))
    .addBands(valid_gedi)
)

training_samples = (
    training_stack
    .stratifiedSample(
        numPoints=3000,
        classBand="valid_gedi",
        region=training_region,
        scale=25,
        classValues=[1],
        classPoints=[3000],
        seed=42,
        dropNulls=True,
    )
    .filter(ee.Filter.eq("valid_gedi", 1))
)

model = (
    ee.Classifier.smileRandomForest(
        numberOfTrees=300,
        minLeafPopulation=2,
        seed=42,
    )
    .setOutputMode("REGRESSION")
    .train(
        features=training_samples,
        classProperty="agbd",
        inputProperties=feature_bands,
    )
)

forest = (
    ee.Image("ESA/WorldCover/v200/2021")
    .select("Map")
    .eq(10)
    .clip(target_region)
)

predicted_agbd = (
    predictors
    .select(feature_bands)
    .clip(target_region)
    .classify(model)
    .rename("Predicted_AGBD_Mg_ha")
    .updateMask(forest)
)

mean_agbd = predicted_agbd.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=target_region,
    scale=25,
    maxPixels=1e9,
).get("Predicted_AGBD_Mg_ha").getInfo()

print(f"Prototype mean predicted AGBD: {mean_agbd:.2f} Mg/ha")
print(f"Prototype mean carbon density: {mean_agbd * 0.47:.2f} Mg C/ha")

Map = geemap.Map(center=[-3.17613, -60.16615], zoom=12)

Map.addLayer(
    s2_full.clip(target_region),
    {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000},
    "Sentinel-2 true color",
)
Map.addLayer(
    predicted_agbd,
    {"min": 0, "max": 400, "palette": ["yellow", "orange", "darkgreen"]},
    "Prototype predicted AGBD (Mg/ha)",
)
Map.addLayer(target_region, {"color": "cyan"}, "Target boundary")

Map.to_html("live_agbd_inference.html")
print("Created live_agbd_inference.html")