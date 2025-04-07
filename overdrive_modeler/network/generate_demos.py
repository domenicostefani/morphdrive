
import os, numpy as np
os.chdir(os.path.abspath(os.path.dirname(__file__)))
from glob import glob
import pandas as pd

DATAFRAME_PATH = r'C:\Users\cimil\Develop\DAFx25-Pedaliny\ROBOT_RECORDER\overdrive_modeler\network\VAE\4-2025-03-05_23-11_tsne_latents.csv'
assert os.path.exists(DATAFRAME_PATH), f"Path {DATAFRAME_PATH} does not exist"
df = pd.read_csv(DATAFRAME_PATH, sep=',', header=0, index_col=0)
print(df.head())

# python .\inference_gui_static.py i C:\Users\cimil\Develop\DAFx25-Pedaliny\ROBOT_RECORDER\DB_DISK\demos\bass.wav

DB_WET_EXCERPTS_FOLDER = r"C:\Users\cimil\Develop\DAFx25-Pedaliny\ROBOT_RECORDER\DB_DISK\demos\db_inputs\split"
assert os.path.exists(DB_WET_EXCERPTS_FOLDER), f"Path {DB_WET_EXCERPTS_FOLDER} does not exist"

NETWET_EXCERPTS_FOLDER = r"C:\Users\cimil\Develop\DAFx25-Pedaliny\ROBOT_RECORDER\DB_DISK\demos\morphdrive_output"
if not os.path.exists(NETWET_EXCERPTS_FOLDER):
    os.makedirs(NETWET_EXCERPTS_FOLDER)

DRY_FOLDER = r"C:\Users\cimil\Develop\DAFx25-Pedaliny\ROBOT_RECORDER\DB_DISK\demos\DRY"
assert os.path.exists(DRY_FOLDER), f"Path {DRY_FOLDER} does not exist"

wet_db_files = glob(os.path.join(DB_WET_EXCERPTS_FOLDER, "*.wav"))
dry_files = [os.path.basename(a) for a in glob(os.path.join(DRY_FOLDER, "*.wav"))]
# print('\n'.join(wet_db_files))

for wetdbfile in wet_db_files:
    filename = os.path.basename(wetdbfile)
    assert filename[:2] == 'a_', f"File {filename} does not start with 'a_'"
    filenamespl = filename[2:].split('_')
    pedalname = filenamespl[0]
    gain = int(filenamespl[1].lstrip('g'))
    tone = int(filenamespl[2].lstrip('t'))
    # print('>>>>',pedalname, gain, tone)

    perpedal = df.loc[pedalname]
    test = perpedal[(perpedal['gain'] == gain) & (perpedal['tone'] == tone)]
    coords = test['coords'].values[0].strip('[]').split(',')
    coords = [float(s) for s in coords]
    # print('coords', coords)
    
    drysoundname = wetdbfile.split('_')[-1].split('.')[0]+'.wav'
    assert drysoundname in dry_files, f"File {drysoundname} not found in {DRY_FOLDER}"
    # print('drysoundname',drysoundname)
    drysoundpath = os.path.join(DRY_FOLDER, drysoundname)
    assert os.path.exists(drysoundpath), f"Path {drysoundpath} does not exist"

    netwetname = os.path.join(NETWET_EXCERPTS_FOLDER, f"wet_{filename}")

    command = f'python ./inference_gui_static.py -i "{drysoundpath}" -xy {coords[0]} {coords[1]} -o "{netwetname}"'
    print(command)
    os.system(command)
    


    # exit()