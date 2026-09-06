import csv
import matplotlib.pyplot as plt
import ee
import geemap

ee.Initialize(project="composed-arch-476417-e5")

# 50 km study area near Manaus, Amazon rainforest
region = ee.Geometry.Point([-60.025, -3.119]).buffer(50000)


def mask_clouds(image):
    scl = image.select("SCL")

    clear_pixels = (
        scl.neq(3)   # Cloud shadow
        .And(scl.neq(8))   # Medium-probability cloud
        .And(scl.neq(9))   # High-probability cloud
        .And(scl.neq(10))  # Cirrus cloud
    )

    return image.updateMask(clear_pixels)


def make_ndvi(year):
    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(f"{year}-06-01", f"{year}-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_clouds)
        .median()
        .clip(region)
    )

    return image.normalizedDifference(["B8", "B4"]).rename("NDVI")


# Use tree-cover pixels only.
forest = (
    ee.Image("ESA/WorldCover/v200/2021")
    .select("Map")
    .eq(10)
    .clip(region)
)

ndvi_2023 = make_ndvi(2023).updateMask(forest)
ndvi_2024 = make_ndvi(2024).updateMask(forest)

change = ndvi_2024.subtract(ndvi_2023).rename("NDVI_change")
strong_loss = change.lt(-0.10).rename("strong_loss").selfMask()

loss_area_m2 = (
    strong_loss.multiply(ee.Image.pixelArea())
    .reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=30,
        maxPixels=1e10,
    )
    .get("strong_loss")
    .getInfo()
)

mean_2023 = (
    ndvi_2023.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=500,
        maxPixels=1e9,
    )
    .get("NDVI")
    .getInfo()
)

mean_2024 = (
    ndvi_2024.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=500,
        maxPixels=1e9,
    )
    .get("NDVI")
    .getInfo()
)

print(f"Average NDVI in 2023: {mean_2023:.3f}")
print(f"Average NDVI in 2024: {mean_2024:.3f}")
plt.figure(figsize=(6, 4))

plt.bar(
    ["2023", "2024"],
    [mean_2023, mean_2024],
    color=["forestgreen", "darkorange"],
)

plt.ylim(0, 1)
plt.ylabel("Average forest NDVI")
plt.title("Amazon forest vegetation health\nDry season comparison")
plt.savefig("amazon_ndvi_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

print("Created amazon_ndvi_comparison.png")
print(f"Change: {mean_2024 - mean_2023:.3f}")
print(f"Strong vegetation-loss area: {loss_area_m2 / 1_000_000:.2f} km²")

# Select 10 confirmed strong-loss locations.
loss_points = (
    strong_loss.unmask(0)
    .toByte()
    .stratifiedSample(
        numPoints=10,
        classBand="strong_loss",
        region=region,
        scale=30,
        seed=42,
        classValues=[1],
        classPoints=[10],
        geometries=True,
    )
    .getInfo()
)

print("\nSample strong-loss locations:")
for feature in loss_points["features"]:
    if feature["properties"]["strong_loss"] != 1:
        continue

    longitude, latitude = feature["geometry"]["coordinates"]
    print(f"Latitude: {latitude:.5f}, Longitude: {longitude:.5f}")
Map = geemap.Map(center=[-3.119, -60.025], zoom=8)

ndvi_colors = {
    "min": -0.2,
    "max": 0.8,
    "palette": ["brown", "yellow", "lightgreen", "darkgreen"],
}

Map.addLayer(ndvi_2023, ndvi_colors, "NDVI 2023")
Map.addLayer(ndvi_2024, ndvi_colors, "NDVI 2024")
Map.addLayer(
    change,
    {"min": -0.3, "max": 0.3, "palette": ["red", "white", "green"]},
    "Vegetation change: red = loss, green = gain",
)
Map.addLayer(
    strong_loss,
    {"palette": ["red"]},
    "Strong vegetation loss (NDVI drop > 0.10)",
)

Map.to_html("ndvi_comparison.html")
print("Created ndvi_comparison.html")
with open("amazon_loss_hotspots.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["latitude", "longitude"])

    for feature in loss_points["features"]:
        if feature["properties"]["strong_loss"] != 1:
            continue

        longitude, latitude = feature["geometry"]["coordinates"]
        writer.writerow([latitude, longitude])

print("Created amazon_loss_hotspots.csv")