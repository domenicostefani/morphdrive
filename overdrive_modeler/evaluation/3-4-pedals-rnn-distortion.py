import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import os
import soundfile as sf
import pandas as pd
from distortionMetrics import calculate_improved_distortion_metrics, plot_heatmaps
os.chdir(os.path.dirname(os.path.abspath(__file__)))



##
#
#  Numero di colonne e righe nel plot
#
#
GRID_COLS = 100

SAVETEMP = True # Salva i file temporanei


path = '../network'
assert os.path.exists(path), "Path not found at %s" % os.path.abspath(path)
if not os.path.exists(os.path.basename(path)):
    os.symlink(path, os.path.basename(path))


from network.inference_gui_static import HyperdriveStaticRNNinference, gen_sine

## Small model
# PATH_TO_TSNE_DATAFRAME = './network/VAE/4-PAPER_2025-03-05_23-11_tsne_latents.csv'
# MLP_MODEL_PATH = './network/MLP/4-PAPER_mlp.pth'
# RNN_MODEL_PATH = './network/RNNprocessingNetwork/4-PAPER_2025-03-05_static_small.pth'

## Large static model
PATH_TO_TSNE_DATAFRAME = './network/VAE/4-PAPER_2025-03-05_23-11_tsne_latents.csv'
MLP_MODEL_PATH = './network/MLP/4-PAPER_mlp.pth'
RNN_MODEL_PATH = './network/RNNprocessingNetwork/4-PAPER_2025-03-05_static.pth'
SR = 48000





assert os.path.exists(PATH_TO_TSNE_DATAFRAME), "Path not found at %s" % os.path.abspath(PATH_TO_TSNE_DATAFRAME)
assert os.path.exists(MLP_MODEL_PATH), "Path not found at %s" % os.path.abspath(MLP_MODEL_PATH)
assert os.path.exists(RNN_MODEL_PATH), "Path not found at %s" % os.path.abspath(RNN_MODEL_PATH)
assert os.path.splitext(RNN_MODEL_PATH)[1] == '.pth', "RNN model must be a .pth file"


tsne_dataframe = pd.read_csv(PATH_TO_TSNE_DATAFRAME)

xycoords = tsne_dataframe['coords'].values

x,y = [],[]
for coord in xycoords:
    coord = coord.strip(' []')
    x.append(float(coord.split(',')[0]))
    y.append(float(coord.split(',')[1]))

xs = np.linspace(min(x), max(x), GRID_COLS)
ys = np.linspace(min(y), max(y), GRID_COLS)


inferencer = HyperdriveStaticRNNinference(RNN_MODEL_PATH, MLP_MODEL_PATH, 'tsne_dataframe.csv', SR)


# generate a 1 second sine wave at SINE_FREQ Hz, at -6 dB
SINE_FREQ = 440
SINE_AMPLITUDE_DB = -6
SINE_DURATION_SEC = 0.1
SINE_AMPLITUDE = 10**(SINE_AMPLITUDE_DB/20)
clean_signal = gen_sine(SINE_FREQ, SINE_DURATION_SEC, SR) * SINE_AMPLITUDE


OUTPUT_DIR = f"./tmp/{GRID_COLS}grid_xydistortion{os.path.basename(RNN_MODEL_PATH).replace('.pth','')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if SAVETEMP:
    sf.write(os.path.join(OUTPUT_DIR, 'sine_input.wav'), clean_signal, SR)


# Initialize results matrices
results_harmonic =   np.zeros((len(xs), len(ys)))
results_crest =      np.zeros((len(xs), len(ys)))
results_perceptual = np.zeros((len(xs), len(ys)))
results_thd =        np.zeros((len(xs), len(ys)))

for ix,x in enumerate(xs):
    for iy,y in enumerate(ys):
        print(f'Processing {ix+1}/{len(xs)} x {iy+1}/{len(ys)}', end='\r', flush=True)
        # pass
        rnn_output = inferencer.inference_memoryless(clean_signal, x, y)
        if SAVETEMP:
            sf.write(os.path.join(OUTPUT_DIR, f'x{x:.2f}y{y:.2f}.wav'), rnn_output, SR)

        metrics = calculate_improved_distortion_metrics(clean_signal, rnn_output)
            
        # Store results
        results_harmonic[ix, iy] = metrics['harmonic_ratio']
        results_crest[ix, iy] = metrics['crest_factor_ratio']
        results_perceptual[ix, iy] = metrics['perceptual_diff']
        results_thd[ix, iy] = metrics['thd']

argmax_perceptual = np.unravel_index(np.argmax(results_perceptual), results_perceptual.shape)
argmin_perceptual = np.unravel_index(np.argmin(results_perceptual), results_perceptual.shape)
argmean_perceptual = np.unravel_index(np.argmin(np.abs(results_perceptual - np.mean(results_perceptual))), results_perceptual.shape)

results_harmonic_df = pd.DataFrame(results_harmonic, index=xs, columns=ys)
results_crest_df = pd.DataFrame(results_crest, index=xs, columns=ys)
results_perceptual_df = pd.DataFrame(results_perceptual, index=xs, columns=ys)
results_thd_df = pd.DataFrame(results_thd, index=xs, columns=ys)

import datetime
now = datetime.datetime.now()
results_harmonic_df.to_csv(os.path.join(OUTPUT_DIR,f'results_harmonic_{now.strftime("%Y-%m-%d_%H-%M")}.csv'))
results_crest_df.to_csv(os.path.join(OUTPUT_DIR,f'results_crest_{now.strftime("%Y-%m-%d_%H-%M")}.csv'))
results_perceptual_df.to_csv(os.path.join(OUTPUT_DIR,f'results_perceptual_{now.strftime("%Y-%m-%d_%H-%M")}.csv'))
results_thd_df.to_csv(os.path.join(OUTPUT_DIR,f'results_thd_{now.strftime("%Y-%m-%d_%H-%M")}.csv'))

plot_heatmaps(results_harmonic, results_crest, results_perceptual, xs, ys, savepath=os.path.join(OUTPUT_DIR,f'heatmaps_{now.strftime("%Y-%m-%d_%H-%M")}.png'),
              results_thd=results_thd)

print('Done!', flush=True)