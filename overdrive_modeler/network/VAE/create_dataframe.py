import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory

DATASET_DIR = '../dataset_32000Hz/'
OUTPUT_PATH = os.path.join(DATASET_DIR,'pedals_dataframe.csv')

data = []

for pedal_name in os.listdir(DATASET_DIR):
    pedal_path = os.path.join(DATASET_DIR, pedal_name)

    if not os.path.isdir(pedal_path):
        continue

    print('Processing', pedal_name)

    file_dict = {}
    
    pedal_files = os.listdir(pedal_path)
    pedal_files = [file for file in pedal_files if file.endswith('.wav')]

    for pfidx,file_name in enumerate(os.listdir(pedal_path)):
        print(f"Processing {pfidx+1}/{len(pedal_files)}", end='\r')
        parts = file_name.split("_")

        file_type = parts[0]
        g_value = parts[2][1]  
        t_value = parts[3][1]  

        complete_name = f"{pedal_name}_g{g_value}_t{t_value}"
        file_path = os.path.join(pedal_path, file_name)
        file_path = os.path.abspath(file_path)

        if complete_name not in file_dict:
            file_dict[complete_name] = {
                "pedal_name": pedal_name,
                "g_value": g_value,
                "t_value": t_value,
                "audio_path": "",
                "sweep_path1": "",
                "sweep_path2": "",
                "sweep_path3": "",
                "noise_path": ""
            }

        if file_type == "a":
            file_dict[complete_name]["audio_path"] = file_path
        elif file_type == "s0":
            file_dict[complete_name]["sweep_path1"] = file_path
        elif file_type == "s1":
            file_dict[complete_name]["sweep_path2"] = file_path
        elif file_type == "s2":
            file_dict[complete_name]["sweep_path3"] = file_path
        elif file_type == "n":
            file_dict[complete_name]["noise_path"] = file_path

    print('file_dict contains', len(file_dict), 'files')
    data.extend(file_dict.values())

df = pd.DataFrame(data)
df.insert(0, "complete_name", df.index)

df.to_csv(OUTPUT_PATH, index=False)
print(f"Dataframe saved to {OUTPUT_PATH}")