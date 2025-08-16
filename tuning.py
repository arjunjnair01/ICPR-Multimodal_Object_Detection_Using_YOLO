import numpy as np
from ultralytics import YOLO

# Initialize the model
model = YOLO("path_to_model")

# Define the ranges for IoU and confidence thresholds
iou_values = np.arange(0.3, 0.9, 0.1)  # IoU from 0.3 to 0.9 with a step of 0.1
conf_values = np.arange(0.1, 0.9, 0.1)  # Confidence from 0.1 to 0.9 with a step of 0.1

best_map = 0
best_iou, best_conf = 0, 0

# Path to your dataset YAML file
dataset_path = 'path_to_dataset'

# Perform the grid search
for iou in iou_values:
    for conf in conf_values:
        # Validate the model using the current IoU and confidence thresholds
        results = model.val(data=dataset_path, iou=iou, conf=conf,imgsz=1280)
        
        # Access mAP using the correct attribute (map)
        current_map = results.box.map    # Mean AP at IoU thresholds from 0.5 to 0.95
        
        print(f"IoU: {iou:.2f}, Conf: {conf:.2f}, mAP50-95: {current_map:.4f}")

        # Track the best performing thresholds
        if current_map > best_map:
            best_map = current_map
            best_iou = iou
            best_conf = conf

print("\nGrid Search Results:")
print(f"Best IoU threshold: {best_iou:.2f}")
print(f"Best confidence threshold: {best_conf:.2f}")
print(f"Best mAP50-95: {best_map:.4f}")

# Save the results to a file
with open('grid_search_results.txt', 'w') as f:
    f.write(f"Best IoU threshold: {best_iou:.2f}\n")
    f.write(f"Best confidence threshold: {best_conf:.2f}\n")
    f.write(f"Best mAP50-95: {best_map:.4f}\n")