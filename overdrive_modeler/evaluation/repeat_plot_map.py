import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import pandas as pd
import os
import soundfile as sf
from distortionMetrics import calculate_improved_distortion_metrics, plot_heatmaps
os.chdir(os.path.dirname(os.path.abspath(__file__)))



##
#
#  Numero di colonne e righe nel plot
#
#
SR=48000
path = 'tmp/100grid_xydistortion4-PAPER_2025-03-05_static'
path = 'tmp/10grid_xydistortion4-PAPER_2025-03-05_static'
OUTPUT_DIR = 'replot'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_SETTING_POINTS = True
TSNE_DATAFRAME_PATH = '../network/VAE/4-PAPER_2025-03-05_23-11_tsne_latents.csv'




if PLOT_SETTING_POINTS:
    assert os.path.exists(TSNE_DATAFRAME_PATH), "Path not found at %s" % os.path.abspath(TSNE_DATAFRAME_PATH)
    assert os.path.splitext(TSNE_DATAFRAME_PATH)[1] == '.csv', "TSNE dataframe must be a .csv file"
    coords_labels = pd.read_csv(TSNE_DATAFRAME_PATH)
    # Take only columns 'label_name' and 'coords'
    coords_labels = coords_labels[['label_name', 'coords']]
    # coords is a string of the form '[floatx,floaty]', extract two float columns 'x' and 'y'
    coords_labels['x'] = coords_labels['coords'].apply(lambda x: float(x.strip(' []').split(',')[0]))
    coords_labels['y'] = coords_labels['coords'].apply(lambda x: float(x.strip(' []').split(',')[1]))
    # Drop the 'coords' column
    coords_labels.drop(columns=['coords'], inplace=True)


assert os.path.exists(path), "Path not found at %s" % os.path.abspath(path)
assert os.path.isdir(path), "Path is not a directory at %s" % os.path.abspath(path)

from glob import glob
audiofiles = glob(os.path.join(path, 'x*y*.wav'))
get_x = lambda f: float(os.path.basename(f).split('y')[0].lstrip('x'))
get_y = lambda f: float(os.path.basename(f).split('y')[1].split('.wav')[0])

# xs = sorted(list(set([float(os.path.basename(f).split('y')[0].lstrip('x')) for f in audiofiles])))
# ys = sorted(list(set([float(os.path.basename(f).split('y')[1].split('.wav')[0]) for f in audiofiles])))

xs = sorted(list(set([get_x(f) for f in audiofiles])))
ys = sorted(list(set([get_y(f) for f in audiofiles])))
gridsize = len(xs) 
assert len(xs) == len(ys), f"Number of x and y values must be equal: {len(xs)} != {len(ys)}"

clean_signal_filename = os.path.join(path, 'sine_input.wav')
assert os.path.exists(clean_signal_filename), f"File not found at {os.path.abspath(clean_signal_filename)}"
print(f'Loading clean signal from {clean_signal_filename}')
clean_signal, sr = sf.read(clean_signal_filename)
assert sr == SR, f"Sample rate mismatch: {sr} != {SR}"

# Initialize results matrices
results_harmonic =   np.zeros((len(xs), len(ys)))
results_crest =      np.zeros((len(xs), len(ys)))
results_perceptual = np.zeros((len(xs), len(ys)))
results_thd =        np.zeros((len(xs), len(ys)))
results_speccent =  np.zeros((len(xs), len(ys)))

all_files = [(get_x(file),get_y(file),file) for file in audiofiles]
all_files.sort()

for idx, (x, y,rnn_output_filename) in enumerate(all_files):
    # print(f'Processing {ix+1}/{len(xs)} x {iy+1}/{len(ys)}', end='\r')
    assert os.path.exists(rnn_output_filename), f"File not found at {os.path.abspath(rnn_output_filename)}"
    rnn_output, sr = sf.read(rnn_output_filename)
    assert sr == SR, f"Sample rate mismatch: {sr} != {SR}"

    metrics = calculate_improved_distortion_metrics(clean_signal, rnn_output)

    

    ix = int(round((gridsize-1)*x,0))
    iy = int(round((gridsize-1)*y,0))
    print(f'Processing {idx+1}/{len(all_files)}: x={x:.2f} y={y:.2f} -> ix={ix} iy={iy}')

    # if x == 0.89 or x == 0.84 or x == 0.95:
    #     print('x',x)
    #     print('ix',ix)
    # if x == 0.95:
    #     exit()

    # Store results
    results_harmonic[ix, iy] = metrics['harmonic_ratio']
    results_crest[ix, iy] = metrics['crest_factor_ratio']
    results_perceptual[ix, iy] = metrics['perceptual_diff']
    results_thd[ix, iy] = metrics['thd']
    if metrics['thd'] == 0:
        print(f'Warning: THD is 0 for x={x:.2f} y={y:.2f}')
    if metrics['thd'] >= 17.0:
        print(f'Warning: THD is >= 19.0 for x={x:.2f} y={y:.2f}')
    results_speccent[ix, iy] = metrics['spectral_centroid']


argmax_perceptual = np.unravel_index(np.argmax(results_perceptual), results_perceptual.shape)
argmin_perceptual = np.unravel_index(np.argmin(results_perceptual), results_perceptual.shape)
argmean_perceptual = np.unravel_index(np.argmin(np.abs(results_perceptual - np.mean(results_perceptual))), results_perceptual.shape)

results_harmonic_df = pd.DataFrame(results_harmonic, index=xs, columns=ys)
results_crest_df = pd.DataFrame(results_crest, index=xs, columns=ys)
results_perceptual_df = pd.DataFrame(results_perceptual, index=xs, columns=ys)

import datetime
now = datetime.datetime.now()
# results_harmonic_df.to_csv(os.path.join(OUTPUT_DIR,f'results_harmonic_{now.strftime("%Y-%m-%d_%H-%M")}.csv'))
# results_crest_df.to_csv(os.path.join(OUTPUT_DIR,f'results_crest_{now.strftime("%Y-%m-%d_%H-%M")}.csv'))
# results_perceptual_df.to_csv(os.path.join(OUTPUT_DIR,f'results_perceptual_{now.strftime("%Y-%m-%d_%H-%M")}.csv'))
results_speccent_df = pd.DataFrame(results_speccent, index=xs, columns=ys)

plot_heatmaps(results_harmonic, results_crest, results_perceptual, xs, ys, savepath=os.path.join(OUTPUT_DIR,f'heatmaps_{now.strftime("%Y-%m-%d_%H-%M")}.png'), results_thd = results_thd,
              results_speccentroid=results_speccent,
              setting_points = coords_labels if PLOT_SETTING_POINTS else None,
              axlabels=('',''))

print('Done!', flush=True)