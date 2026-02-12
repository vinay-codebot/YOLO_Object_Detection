import torch
import cv2
from ultralytics import YOLO

print("Pytorch version:", torch.__version__)
print("Opencv version:", cv2.__version__)

model= YOLO("yolov8n.pt")
print("YOLO Loaded sucessfully!")