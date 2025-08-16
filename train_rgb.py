#These are not the actual training scripts. This given code file is just a sample.
from ultralytics import YOLO

#rgb
modelR = YOLO("yolov8s.pt")  # Load the pretrained model
modelR.train(data="./rgb.yaml", epochs=10, batch=8,imgsz=640,)
