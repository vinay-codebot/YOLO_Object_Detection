# YOLOv8 Object Detection Project

## Overview
- This project focuses on comparing YOLOv8 models for object detection in vedio data. 
  Using a sample video, we analyzed the accuracy, speed, and usability of both models based on frame-by-frame detections and confidence scores.
- The goal was to:
 - Evaluate model accuracy using confidence scores per frame.
 - Identify anomalies—frames with low confidence or inconsistent detections.
 - Assess speed and practical usability for real-time video processing.
 - Provide visual and numerical insights for model selection

## Day 1 Progress
- Python virtual environment setup completed
- Required libraries installed: PyTorch, OpenCV, Ultralytics YOLOv8
- YOLOv8 pre-trained model downloaded (yolov8n.pt)
- Detection works on a sample image (`sample.jpg`)
- Detection output saved as `sample_detected.jpg`
- ## Screenshot proof 
  ![Detection Result](Day1_Screenshot.png)

## Day 2 progress
- Webcam integration:
  Used cv2.VideoCapture(0) to access the system webcam.
  Ensures that detection is tested on dynamic, real-world input.

- FPS measurement and smoothing:
  Calculated FPS per frame to measure speed: fps = 1 / (current_time - prev_time)
  Applied moving average over last 10 frames → reduces fluctuations and makes FPS more reliable for analysis.

- Logging:
  Logs stored in outputs/logs/logs.txt
  Each log includes timestamp, FPS, number of objects detected, and confidence scores.
  Purpose: Quantitative evidence for research comparison.

-Video recording and screenshots:
  Annotated frames saved to video using cv2.VideoWriter.
  Key frames saved as screenshots using cv2.imwrite.
  Purpose: Provides visual proof of detection performance.

# Day 3 progress
- Logs Parsing:
  - Converted raw logs into CSV files using Python:
    -Extracted min, max, mean, std of confidence per frame.
  - Created a structured DataFrame for analysis:
    df_s = pd.read_csv("yolov8s_log_parsed.csv")
    df_n = pd.read_csv("yolov8n_log_parsed.csv")

- Anomaly Detection:
  - Defined anomalous frames based on thresholds:
     conf_max < 0.5
     conf_mean < 0.4
     conf_std > 0.25
  - These frames indicate low-confidence detections or inconsistent results:
    anomalies_s = df_s[
        (df_s['conf_max'] < 0.5) |
        (df_s['conf_mean'] < 0.4) |
        (df_s['conf_std'] > 0.25)
     ]
  - Observation:
     - Yolo-n has silghtly more anamolies than YOLO-s.
       Which means YOLO-s outperformes YOLO-n in detection stability. YOLO-n priortizes speed over robustness.

- Metrics for evaluation:
    We focused on three major metrics:
     - Accuracy : Confidence of correct detection by taking Mean confidence of objects per frame (conf_mean)
     - Usability/Speed : Practical performance in real-time by taking FPS per
       frame, cumulative usable frames (confidence > threshold & FPS > 15).
  - Contibution of each confidence measures:
      conf_min: Lowest detection confidence in frame; identifies risky detections.
      conf_max: Highest detection confidence; identifies strong detections.
      conf_mean: Average confidence; main metric for accuracy.
      conf_std: Spread of confidences; high value indicates inconsistent
                detection in frame.
  
- Visualization:
 Generated three separate graphs:
  - Accuracy per Frame
     Plotted mean confidence for YOLO-s and YOLO-n.
     Red/Orange dots highlight anomalies.
  - Speed per Frame
     Binary plot showing which frames are anomalous (1 = anomaly, 0 = normal).
     Helps detect patterns where models struggle.
  - Usable Frames
    Frames where confidence > threshold AND FPS > 15.
    Shows real-time usability of models.
                        
- Numerical Comparison Table:
   - Shows tradeoff:
       YOLO-s has higher accuray but it slower.
       YOLO-n is faster,better real-time usability but inconsistent and less 
       accuracy.
  
