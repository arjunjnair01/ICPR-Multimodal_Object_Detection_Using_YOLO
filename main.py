from ultralytics import YOLO
import cv2
import torch
from torchvision.ops import nms
import os

# Load the models
modelR = YOLO("path_to_rgb_weights")  # for RGB
modelD = YOLO("path_to_depth_weights")  # for Depth
modelI = YOLO("path_to_infrared_weights")  # for Infrared

results_dir = .results39
if not os.path.exists(results_dir)
    os.makedirs(results_dir)
#pred
pred = os.path.join(results_dir,pred)
if not os.path.exists(pred)
    os.makedirs(pred)
    
# Define dataset directories
infrared_dir = .datasetsimagestestinfrared
rgb_dir = .datasetsimagestestcolor
depth_dir = .datasetsimagestestdepth

# Get list of images
infrared_images = sorted(os.listdir(infrared_dir))
rgb_images = sorted(os.listdir(rgb_dir))
depth_images = sorted(os.listdir(depth_dir))

# Loop through images
for i in range(len(infrared_images))
    # Load each image
    image_ir = os.path.join(infrared_dir, infrared_images[i])
    image_rgb = os.path.join(rgb_dir, rgb_images[i])
    image_depth = os.path.join(depth_dir, depth_images[i])

    image_name = os.path.basename(image_ir)
    final_name = os.path.splitext(image_name)[0]
    
    # Read RGB image for visualization
    image = cv2.imread(image_rgb)

    # Predict on the image using each model
    resultsR = modelR(image_rgb,iou=0.7,conf=0.5)
    resultsI = modelI(image_ir,iou=0.7,conf=0.5)
    resultsD = modelD(image_depth,iou=0.7,conf=0.5)

    #For submission 1, considering class predicted by RGB
    r_class = resultsR[0].boxes.cls


    # Extract boxes and scores from each result
    boxes_rgb, scores_rgb = resultsR[0].boxes.xyxy, resultsR[0].boxes.conf
    boxes_depth, scores_depth = resultsD[0].boxes.xyxy, resultsD[0].boxes.conf
    boxes_infrared, scores_infrared = resultsI[0].boxes.xyxy, resultsI[0].boxes.conf

    # Concatenate all boxes and scores
    all_boxes = torch.cat((boxes_rgb, boxes_depth, boxes_infrared), dim=0)
    all_scores = torch.cat((scores_rgb, scores_depth, scores_infrared), dim=0)

    # Apply Non-Maximum Suppression (NMS)
    nms_indices = nms(all_boxes, all_scores, iou_threshold=0.3)
    nms_boxes = all_boxes[nms_indices]
    nms_scores = all_scores[nms_indices]

    # Visualize or further process the final detections
    for c,box, score in zip(r_class,nms_boxes, nms_scores)
        if torch.isnan(box).any() or torch.isnan(score)
            print(fSkipping NaN box or score {box}, {score})
            continue

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f{score.2f}, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the final image with detected boxes
    output_path = os.path.join(results_dir, fname_{i+1}.jpg)
    cv2.imwrite(output_path, image)

    print(fProcessed and saved {output_path})


    # Save predictions to a text file
    pred_file = os.path.join(pred, f{final_name}.txt)
    with open(pred_file, 'w') as f
        for c,box, score in zip(r_class, nms_boxes, nms_scores)
            if torch.isnan(box).any() or torch.isnan(score)
                continue
            x1, y1, x2, y2 = box
            x1 = (x1  1920)  672
            x2 = (x2  1920)  672
            y1 = (y1  1080)  384
            y2 = (y2  1080)  384

            f.write(f{x1} {y1} {x2} {y2} {score} {c} n) #[xyxy, conf, cls]

cv2.destroyAllWindows()

