"""
    This script takes the raw dataset files from the PureData recorder patch and normalizes them so that the loudest
    peak for each PEDAL is at -3dB. All the remaining audio files for the same pedal are normalized by the same factor.
    The script also separates the audio recording into the three sine sweeps at different gain, the noise floor recording, and the 
    generic audio input taken from:
     - Yeh, Y. T., Hsiao, W. Y., & Yang, Y. H. (2024). Hyper recurrent neural network: Condition mechanisms for black-box audio effect modeling. arXiv preprint arXiv:2408.04829.  
    Their demo page: https://yytung.notion.site/HyperRNN-ce8cecb3ef6a41fcbcb54a94f9f6de6f
    Their DB download link: https://drive.google.com/file/d/1y3iQH94dAZbRgP33Pt4lJakgolvV7Xal/view
    Their code: https://github.com/ytsrt66589/pyneuralfx/tree/main

    The recordings should be in ../dataset_raw/
"""

import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import os
from glob import glob
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script file location

# Configuration
input_folder = "../dataset_raw"
assert os.path.exists(input_folder) and os.path.isdir(input_folder), "Input folder not found!"
assert len(glob(os.path.join(input_folder, "*.wav"))) > 0, "No WAV files found in input folder!"
output_folder = "../dataset"

UNPROCESSED_INPUT_FILENAME = "0-unprocessed_input.wav"
assert os.path.exists(os.path.join(input_folder, UNPROCESSED_INPUT_FILENAME)), "Input file not found!"
normalization_level_db = -3  # Target normalization level in dB
initial_offset_samples = 1268  # Initial offset in samples
pause_samples = 24000  # 0.5 seconds for 48kHz sampling rate
sweep_length_seconds = 4  # 4 seconds
sampling_rate = 48000  # Assuming 48kHz sampling rate

# Convert ms to samples
initial_noise_offset = int((200 / 1000) * sampling_rate)  # Convert 200ms to samples
noise_length = int((300 / 1000) * sampling_rate)  # Convert 300ms to samples

log_file = os.path.join(output_folder, "normalization_log.csv")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Helper function to calculate the max amplitude in dB
def calculate_max_amplitude(audio):
    return 20 * np.log10(np.max(np.abs(audio)))

# Helper function to normalize audio
def normalize_audio(audio, target_db, max_db):
    adjustment = target_db - max_db
    gain_factor = 10 ** (adjustment / 20)
    return audio * gain_factor

# Group files by effect
effect_groups = {}
for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        effect_name = file.split("_")[0] if file != UNPROCESSED_INPUT_FILENAME else 'input' # Extract effect name
        effect_groups.setdefault(effect_name, []).append(file)

# DataFrame to store normalization logs
log_data = []

# Process each effect group
for effect, files in effect_groups.items():
    print(f"Processing effect group: {effect}")

    # Create a subfolder for the effect
    effect_folder = os.path.join(output_folder, effect)
    os.makedirs(effect_folder, exist_ok=True)

    max_amplitude = float('-inf')
    file_amplitudes = {}

    # Find the max amplitude across all files in the group
    for file in files:
        file_path = os.path.join(input_folder, file)
        audio, sr = librosa.load(file_path, sr=sampling_rate)
        file_max = calculate_max_amplitude(audio)
        file_amplitudes[file] = file_max
        max_amplitude = max(max_amplitude, file_max)

    # Normalize and log each file, then create chunks directly
    for file in files:
        file_path = os.path.join(input_folder, file)
        audio, sr = librosa.load(file_path, sr=sampling_rate)
        if not effect == 'input': # Skip normalization for the input file
            normalized_audio = normalize_audio(audio, normalization_level_db, max_amplitude)

            # Log normalization details in DataFrame
            normalized_level = calculate_max_amplitude(normalized_audio)
            log_data.append([file, file_amplitudes[file], normalized_level])
            print(f"Normalized {file}: {file_amplitudes[file]:.2f} dB -> {normalized_level:.2f} dB")

            # Chunking process
            audio = normalized_audio[initial_offset_samples:]  # Remove initial offset

        chunk_length_samples = sweep_length_seconds * sr

        # Create and save chunks
        for i in range(3):  # First 3 chunks (s0, s1, s2)
            start = i * (chunk_length_samples + pause_samples)
            end = start + chunk_length_samples
            chunk = audio[start:end]
            chunk_name = f"s{i}_{file}"
            chunk_path = os.path.join(effect_folder, chunk_name)  # Save in effect's subfolder
            sf.write(chunk_path, chunk, sr, subtype="PCM_24")
            print(f"Saved S chunk: {chunk_name}")

        # Extract noise floor 
        start = 3 * chunk_length_samples + 2 * pause_samples + initial_noise_offset
        end = start + noise_length

        if start < len(audio) and end <= len(audio):  # Ensure indices are within bounds
            chunk = audio[start:end]
            chunk_name = f"n_{file}"
            chunk_path = os.path.join(effect_folder, chunk_name)  # Save in effect's subfolder
            sf.write(chunk_path, chunk, sr, subtype="PCM_24")
            print(f"Saved N chunk: {chunk_name}")
        else:
            print(f"Skipping noise chunk for {file} (indices out of range)")

        # Remaining audio as the fourth chunk
        start = 3 * (chunk_length_samples + pause_samples)
        remaining_audio = audio[start:]
        chunk_name = f"a_{file}"
        chunk_path = os.path.join(effect_folder, chunk_name)  # Save in effect's subfolder
        sf.write(chunk_path, remaining_audio, sr, subtype="PCM_24")
        print(f"Saved A chunk: {chunk_name}")

# Save normalization log to CSV
df = pd.DataFrame(log_data, columns=["name", "file_db", "norm_db"])
df['appied_gain_db'] = df['norm_db'] - df['file_db']
df.to_csv(log_file, index=False, float_format="%.2f")

print(f"Processing complete! Normalization log saved at \"{log_file}\".")

df['pedal'] = df['name'].apply(lambda x: x.split("_")[0])
df['gain'] = df['name'].apply(lambda x: int(x.split("_")[1].lstrip("g")))
df['tone'] = df['name'].apply(lambda x: int(x.split("_")[2].lstrip("t").rstrip(".wav")))
# Drop the 'name' column
df.drop(columns=['name'], inplace=True)

print(df)

# Compute summary statistics for df
summary = df.groupby('pedal').agg(['mean', 'std', 'max'])
# rename to gain_meain, gain_std etc
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary.reset_index(inplace=True)


print(summary)



for pedal in summary['pedal']:
    if pedal == '0-input':
        continue
    pedal_df = df[df['pedal'] == pedal]
    assert pedal_df['gain'].nunique() == 6, f"Pedal {pedal} is missing some gain values"
    assert pedal_df['tone'].nunique() == 6, f"Pedal {pedal} is missing some tone values"


summary.drop(columns=['gain_mean','gain_std','gain_max','tone_mean','tone_std','tone_max'], inplace=True)


summary_path = os.path.join(output_folder, "normalization_summary.csv")
summary.to_csv(summary_path, float_format="%.2f")
print(f"Summary statistics saved at \"{summary_path}\".")