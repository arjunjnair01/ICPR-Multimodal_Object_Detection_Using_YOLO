#These are not the actual training scripts. This given code file is just a sample.
from ultralytics import YOLO

#depth
modelD = YOLO("custom_yolov8s.yaml")  # Load the pretrained model
modelD.train(data="./depth.yaml", epochs=25, batch=8)