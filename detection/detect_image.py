from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

results = model("sample.jpg")
annotated_frame=results[0].plot()
cv2.imwrite("sample_detected.jpg",annotated_frame)
print("detection completed! output saved as sample_detected.jpg")