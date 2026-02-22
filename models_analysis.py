import pandas as pd
import matplotlib.pyplot as plt


df_s = pd.read_csv("outputs/logs/yolov8s_log_parsed.csv")
df_n = pd.read_csv("outputs/logs/yolov8n_log_parsed.csv")



def find_anomalies(df, max_thresh, mean_thresh, std_thresh):
    return df[
        (df['conf_max'] < max_thresh) |
        (df['conf_mean'] < mean_thresh) |
        (df['conf_std'] > std_thresh)
    ]


anomalies_s = find_anomalies(df_s, 0.5, 0.4, 0.25)
anomalies_n = find_anomalies(df_n, 0.5, 0.4, 0.25)


print(f"YOLO-s anomalies: {len(anomalies_s)} frames")
print(f"YOLO-n anomalies: {len(anomalies_n)} frames")

window = 10  
df_s['FPS_smooth'] = df_s['FPS'].rolling(window).mean()
df_n['FPS_smooth'] = df_n['FPS'].rolling(window).mean()


df_s['conf_mean_smooth'] = df_s['conf_mean'].rolling(window).mean()
df_n['conf_mean_smooth'] = df_n['conf_mean'].rolling(window).mean()




plt.figure(figsize=(12,6))
plt.plot(df_s['conf_mean_smooth'], label='YOLO-s mean confidence (smoothed)')
plt.plot(df_n['conf_mean_smooth'], label='YOLO-n mean confidence (smoothed)')
plt.xlabel("Frame")
plt.ylabel("Mean Confidence (Accuracy)")
plt.title("Smoothed Accuracy Comparison")
plt.legend()
plt.show()  


plt.figure(figsize=(12,6))
plt.plot(df_s['conf_mean_smooth'], label='YOLO-s Mean Confidence', color='blue')
plt.plot(df_n['conf_mean_smooth'], label='YOLO-n Mean Confidence', color='green')
plt.scatter(anomalies_s.index, anomalies_s['conf_mean'], color='red', label='YOLO-s Anomalies')
plt.scatter(anomalies_n.index, anomalies_n['conf_mean'], color='orange', label='YOLO-n Anomalies')
plt.xlabel("Frame")
plt.ylabel("Mean Confidence")
plt.title("Accuracy per Frame (Mean Confidence)")
plt.legend()
plt.show()


plt.figure(figsize=(12,4))
plt.plot(df_s['FPS_smooth'], label='YOLO-s FPS (smoothed)')
plt.plot(df_n['FPS_smooth'], label='YOLO-n FPS (smoothed)')
plt.xlabel("Frame")
plt.ylabel("FPS (Speed)")
plt.title("Smoothed Speed Comparison")
plt.legend()
plt.show()  


plt.figure(figsize=(8,6))
plt.scatter(df_s['conf_mean_smooth'], df_s['FPS_smooth'], label='YOLO-s', alpha=0.5)
plt.scatter(df_n['conf_mean_smooth'], df_n['FPS_smooth'], label='YOLO-n', alpha=0.5)
plt.xlabel("Mean Confidence (Accuracy)")
plt.ylabel("FPS (Speed)")
plt.title("Accuracy vs Speed (Smoothed)")
plt.legend()
plt.show()





# Thresholds for usability
conf_thresh = 0.5
fps_thresh = 15

# Accuracy metrics
accuracy_s = {
    'Mean Conf': df_s['conf_mean'].mean(),
    'Max Conf': df_s['conf_max'].max(),
    'Std Conf': df_s['conf_std'].mean(),
    'Anomaly Frames': len(anomalies_s)
}
accuracy_n = {
    'Mean Conf': df_n['conf_mean'].mean(),
    'Max Conf': df_n['conf_max'].max(),
    'Std Conf': df_n['conf_std'].mean(),
    'Anomaly Frames': len(anomalies_n)
}

# Speed metrics
speed_s = {
    'Avg FPS': df_s['FPS'].mean(),
    'Max FPS': df_s['FPS'].max()
}
speed_n = {
    'Avg FPS': df_n['FPS'].mean(),
    'Max FPS': df_n['FPS'].max()
}

# Usability metrics
usable_s = df_s[(df_s['conf_mean'] >= conf_thresh) & (df_s['FPS'] >= fps_thresh)].shape[0]
usable_n = df_n[(df_n['conf_mean'] >= conf_thresh) & (df_n['FPS'] >= fps_thresh)].shape[0]
usability_s = {'Usable Frames': usable_s}
usability_n = {'Usable Frames': usable_n}




# Combine all metrics into a single DataFrame
metrics_table = pd.DataFrame({
    'YOLO-s': {**accuracy_s, **speed_s, **usability_s},
    'YOLO-n': {**accuracy_n, **speed_n, **usability_n}
})

print(metrics_table)