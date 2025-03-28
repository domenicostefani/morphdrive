"""
    Resample the dataset to 32000Hz for the VAE "controller" model
"""

import os
import librosa
import soundfile as sf
from glob import glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))

audio_files = glob('./*/s0*.wav') # Resample only the first sweep to save space
print(f"Found {len(audio_files)} audio files")

ORIGINAL_SR = 48000
NEW_SR = 32000

OUTPUT_DIR = f'../dataset_{NEW_SR//1000}'
assert os.path.exists(OUTPUT_DIR), f"Output directory {OUTPUT_DIR} does not exist"
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

for iaf, audio_file in enumerate(audio_files):
    new_path = os.path.join(OUTPUT_DIR, os.path.relpath(audio_file, '.'))
    print(f"Resampling {audio_file} to {new_path} [{iaf+1}/{len(audio_files)}]", end='\r')

    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    
    audio, _ = librosa.load(audio_file, sr=ORIGINAL_SR)
    audio_res = librosa.resample(y=audio, orig_sr=ORIGINAL_SR, target_sr=NEW_SR, res_type='soxr_vhq')
    sf.write(new_path, audio_res, NEW_SR)

print("Done")


