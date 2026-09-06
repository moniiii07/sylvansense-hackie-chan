import csv
import json
import ee

ee.Initialize(project="composed-arch-476417-e5")

with open("target_forest.geojson") as file:
    region = ee.Geometry(json.load(file))

# Large regional area for biomass-model training.
training_region = region.centroid(maxError=1).buffer(100000)


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


# Sentinel-2 optical predictor bands.
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(training_region)
    .filterDate("2024-06-01", "2024-09-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    .map(mask_s2_clouds)
    .median()
    .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    .clip(training_region)
)

ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")

# Sentinel-1 radar predictor bands.
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

# GEDI LiDAR-derived AGB density labels, Mg/ha.
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

predictors = s2.addBands(ndvi).addBands(s1).addBands(vv_minus_vh)

# Mark pixels that have a valid GEDI biomass label.
valid_gedi = agbd.mask().rename("valid_gedi").unmask(0).toByte()

# Stratified sampling forces the selection of GEDI-labelled pixels.
sampling_stack = (
    predictors
    .addBands(agbd.unmask(-9999))
    .addBands(valid_gedi)
)

training_samples = (
    sampling_stack
    .stratifiedSample(
        numPoints=3000,
        classBand="valid_gedi",
        region=training_region,
        scale=25,
        classValues=[1],
        classPoints=[3000],
        seed=42,
        dropNulls=True,
        geometries=True,
    )
    .filter(ee.Filter.eq("valid_gedi", 1))
)

data = training_samples.getInfo()["features"]

if not data:
    print("No GEDI-labelled samples found.")
else:
    columns = [
        "longitude", "latitude",
        "B2", "B3", "B4", "B8", "B11", "B12",
        "NDVI", "VV", "VH", "VV_minus_VH", "agbd",
    ]

    with open("agbd_training_samples.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for feature in data:
            props = feature["properties"]
            longitude, latitude = feature["geometry"]["coordinates"]

            writer.writerow({
                "longitude": longitude,
                "latitude": latitude,
                "B2": props["B2"],
                "B3": props["B3"],
                "B4": props["B4"],
                "B8": props["B8"],
                "B11": props["B11"],
                "B12": props["B12"],
                "NDVI": props["NDVI"],
                "VV": props["VV"],
                "VH": props["VH"],
                "VV_minus_VH": props["VV_minus_VH"],
                "agbd": props["agbd"],
            })

    print(f"Created agbd_training_samples.csv with {len(data)} GEDI-labelled samples.")