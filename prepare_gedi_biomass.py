import json
import ee
import geemap

ee.Initialize(project="composed-arch-476417-e5")

with open("target_forest.geojson") as file:
    region = ee.Geometry(json.load(file))


def quality_mask(image):
    return (
        image.updateMask(image.select("l4_quality_flag").eq(1))
        .updateMask(image.select("degrade_flag").eq(0))
        .updateMask(image.select("algorithm_run_flag").eq(1))
    )


# GEDI LiDAR aboveground-biomass-density observations, in Mg/ha.
gedi_agbd = (
    ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY")
    .filterBounds(region)
    .filterDate("2019-04-01", "2024-11-30")
    .map(quality_mask)
    .select("agbd")
    .mean()
    .clip(region)
)

stats = gedi_agbd.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=region,
    scale=25,
    maxPixels=1e9,
).getInfo()

average_agbd = stats.get("agbd")

if average_agbd is None:
    print("No high-quality GEDI biomass observations were found in this small target polygon.")
else:
    carbon_density = average_agbd * 0.47
    print(f"Average GEDI AGBD: {average_agbd:.2f} Mg/ha")
    print(f"Estimated carbon density: {carbon_density:.2f} Mg C/ha")

Map = geemap.Map(center=[-3.17613, -60.16615], zoom=12)

Map.addLayer(
    gedi_agbd,
    {"min": 0, "max": 400, "palette": ["yellow", "orange", "darkgreen"]},
    "GEDI aboveground biomass density (Mg/ha)",
)
Map.addLayer(region, {"color": "cyan"}, "Target forest boundary")

Map.to_html("gedi_biomass.html")
print("Created gedi_biomass.html")