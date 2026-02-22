import pandas as pd
import ast 

log_file="outputs/logs/logs(yv8n).txt"

timestamps=[]
fps_list=[]
num_obj_list=[]
confidences_list=[]

with open(log_file,"r") as f:
    for line in f:
        try:
            parts = line.strip().split('|')
            timestamp = parts[0].strip()
            timestamps.append(timestamp)

            fps = float(parts[1].split(":")[1].strip())
            fps_list.append(fps)

            num_objects = int(parts[2].split(":")[1].strip())
            num_obj_list.append(num_objects)

            conf_str = parts[3].split(":", 1)[1].strip()
            conf_list = ast.literal_eval(conf_str)  
            confidences_list.append(conf_list)

        except Exception as e:
            print("Error parsing line:", line)
            print(e)


df = pd.DataFrame({
    "timestamp": timestamps,
    "FPS": fps_list,
    "num_objects": num_obj_list,
    "confidences": confidences_list
})



df['conf_max']= df['confidences'].apply(lambda x: max(x) if x else 0)
df['conf_min']= df['confidences'].apply(lambda x: min(x) if x else 0)
df['conf_mean']= df['confidences'].apply(lambda x: sum(x)/len(x) if x else 0)
df['conf_std']= df['confidences'].apply(lambda x: pd.Series(x).std() if x else 0)

df.to_csv("outputs/logs/yolov8n_log_parsed.csv", index=False)





