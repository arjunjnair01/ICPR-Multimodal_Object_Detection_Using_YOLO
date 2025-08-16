#These are not the actual training scripts. This given code file is just a sample.
from ultralytics import YOLO

#infrared
modelI = YOLO("custom_yolov8m.yaml")  # Load the pretrained model
modelI.train(data="./infrared.yaml", epochs=10, batch=4)
