from ultralytics import YOLO
import cv2
import time
import datetime

model_name = "yolov8s.pt"  
model = YOLO("yolov8s.pt")  
model.to("cuda")  


cap = cv2.VideoCapture("demo.mp4")
width = int(cap.get(3))
height = int(cap.get(4))
fps_input = cap.get(cv2.CAP_PROP_FPS)


fc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('outputs/videos/Yolov8sDemo.mp4', fc, 20.0, (width, height))

prev_time = time.time()
screenshot_count = 1
fps_list=[]

while True:
    ret , frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    results = model.predict(frame, verbose=False)    
    annotated_frame = results[0].plot()
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_list.append(fps)
    avg_fps = sum(fps_list[-10:]) / len(fps_list[-10:]) 

    num_objects = len(results[0].boxes)
    confidences = results[0].boxes.conf.tolist()  
    with open("outputs/logs/logs(yv8s).txt", "a") as f:
     f.write(f"{datetime.datetime.now()} | FPS: {avg_fps:.2f} | Objects: {num_objects} | Confidences: {confidences}\n")

    

 
    cv2.putText(
        annotated_frame,
        f"{model_name}  FPS: {int(avg_fps)}", 
        (10, 30),                               
        cv2.FONT_HERSHEY_SIMPLEX,       
        0.5,                                    
        (0, 255, 0),                           
        2                                    
    )
    
    cv2.imshow("YOLO(s) Detection", annotated_frame)
    out.write(annotated_frame)

   
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        screenshot_path = f"outputs/screenshots/demo_{screenshot_count}Yolov8s.png"
        cv2.imwrite(screenshot_path, annotated_frame)
        print(f"Screenshot saved: {screenshot_path}")
        screenshot_count += 1
    elif key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
