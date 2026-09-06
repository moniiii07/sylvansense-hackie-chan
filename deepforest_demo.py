from deepforest import get_data, main
from PIL import Image, ImageDraw

model = main.deepforest()
model.load_model(model_name="weecology/deepforest-tree", revision="main")

image_path = get_data("OSBS_029.png")
predictions = model.predict_image(path=image_path)

image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

for _, tree in predictions.iterrows():
    draw.rectangle(
        [tree["xmin"], tree["ymin"], tree["xmax"], tree["ymax"]],
        outline="red",
        width=2,
    )

predictions.to_csv("deepforest_demo_detections.csv", index=False)
image.save("deepforest_demo_detections.png")

print(f"Detected {len(predictions)} tree crowns.")
print("Created deepforest_demo_detections.png")
print("Created deepforest_demo_detections.csv")