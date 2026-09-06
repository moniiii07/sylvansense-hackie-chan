import json
import ee

ee.Initialize(project="composed-arch-476417-e5")

# A 5 km-radius target area around a confirmed vegetation-change location.
target_polygon = ee.Geometry.Point([-60.16615, -3.17613]).buffer(5000)

with open("target_forest.geojson", "w") as file:
    json.dump(target_polygon.getInfo(), file, indent=2)

print("Created target_forest.geojson")
print("Target center: Latitude -3.17613, Longitude -60.16615")