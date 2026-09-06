import os

from deepforest import get_data, main, utilities

model = main.deepforest()
model.load_model(model_name="weecology/deepforest-tree", revision="main")

raster_path = get_data("OSBS_029.tif")

detections = model.predict_tile(
    path=raster_path,
    patch_size=300,
    patch_overlap=0.25,
)

if detections is None or len(detections) == 0:
    print("No tree crowns detected.")
else:
    detections["image_path"] = os.path.basename(raster_path)

    image_detections = utilities.read_file(
        detections,
        root_dir=os.path.dirname(raster_path),
        image_path=os.path.basename(raster_path),
        label="Tree",
    )

    geo_detections = utilities.image_to_geo_coordinates(
        image_detections,
        root_dir=os.path.dirname(raster_path),
    ).to_crs(epsg=4326)

    geo_detections.to_file(
        "deepforest_canopy_detections.geojson",
        driver="GeoJSON",
    )

    print(f"Detected {len(geo_detections)} tree crowns.")
    print("Created deepforest_canopy_detections.geojson")