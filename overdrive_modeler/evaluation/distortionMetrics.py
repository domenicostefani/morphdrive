import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import librosa
import os
import soundfile as sf
from scipy.signal import find_peaks

from network.VAE.utils import DBmetadataRenamer
datasetMetadataRenamer = DBmetadataRenamer()

def get_drysweep(sample_rate=48000):
    """Load the dry sweep from file"""
    sweep, _ = librosa.load(DRY_AUDIO_PATH, sr=sample_rate)
    return sweep


def get_wetsweep(gain=1.0, tone=3.0,sample_rate=48000):
    """Load the wet sweep from file"""
    filepath = os.path.join(WET_AUDIO_FOLDER, f"{SWEEP_NAME}_{PEDAL}_g{int(gain)}_t{int(tone)}.wav")
    assert os.path.exists(filepath), f"Pedal audio file not found at {filepath}"
    audio, _ = librosa.load(filepath, sr=sample_rate)
    return audio

def calculate_thd(original, distorted, sample_rate=48000):
    """
    Calculate Total Harmonic Distortion metric
    
    THD = sqrt(sum of power of harmonics / power of fundamental)
    """
    # Get frequency spectra
    n_fft = 2048
    orig_spectrum = np.abs(np.fft.rfft(original, n=n_fft))
    dist_spectrum = np.abs(np.fft.rfft(distorted, n=n_fft))
    
    
    # Find fundamental frequency (should be around 440 Hz bin)
    freq_bins = np.fft.rfftfreq(len(original), d=1/sample_rate)
    fundamental_idx = np.argmax(orig_spectrum)
    fundamental_freq = freq_bins[fundamental_idx]
    # print('>>>>>> fundamental_freq',fundamental_freq)
    assert fundamental_freq - 440.0 < 10, "Fundamental frequency is not around 440 Hz"
    
    # Calculate THD as ratio of harmonic power to fundamental power
    # First, identify the fundamental and harmonic bins
    fundamental_power = dist_spectrum[fundamental_idx]**2
    
    # Consider harmonics (multiples of fundamental frequency)
    harmonic_powers = 0
    for i in range(2, 10):  # Consider up to 9th harmonic
        harmonic_idx = np.argmin(np.abs(freq_bins - i * fundamental_freq))
        harmonic_powers += dist_spectrum[harmonic_idx]**2
    
    # Calculate THD
    if fundamental_power > 0:
        thd = np.sqrt(harmonic_powers / fundamental_power)
    else:
        thd = 0
    
    return thd


def calculate_improved_distortion_metrics(original, distorted, sample_rate=48000):
    """
    Calculate multiple distortion metrics that correlate better with perceptual distortion
    """
    # Ensure signals are the same length
    min_len = min(len(original), len(distorted))
    original = original[:min_len]
    distorted = distorted[:min_len]
    
    # Normalize levels for fair comparison
    original = original / np.max(np.abs(original))
    distorted = distorted / np.max(np.abs(distorted))
    
    # 1. Harmonic Distortion Analysis
    # --------------------------------
    # Instead of simple energy ratio, analyze specific harmonics
    
    # FFT of both signals
    n_fft = 2048
    hop_length = 512
    
    # Calculate spectrogram
    orig_stft = librosa.stft(original, n_fft=n_fft, hop_length=hop_length)
    dist_stft = librosa.stft(distorted, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to magnitude
    orig_mag = np.abs(orig_stft)
    dist_mag = np.abs(dist_stft)
    
    # Calculate harmonic energy ratio for each frame
    harmonic_energy = []
    fundamental_energy = []
    
    # For each time frame
    for i in range(orig_mag.shape[1]):
        # Find the peak in the original signal (fundamental)
        fund_idx = np.argmax(orig_mag[:, i])
        
        # If the peak is too low, skip this frame
        if orig_mag[fund_idx, i] < 0.1 * np.max(orig_mag):
            continue
            
        # Sum energy of the fundamental in the distorted signal
        fund_energy = dist_mag[fund_idx, i]**2
        fundamental_energy.append(fund_energy)
        
        # Sum energy of the harmonics in the distorted signal
        harm_energy = 0
        for h in range(2, 11):  # 2nd to 10th harmonic
            harm_idx = min(fund_idx * h, dist_mag.shape[0] - 1)
            harm_energy += dist_mag[harm_idx, i]**2
            
        harmonic_energy.append(harm_energy)
    
    # Calculate harmonic-to-fundamental ratio
    if len(fundamental_energy) > 0:
        harm_to_fund_ratio = np.mean(np.array(harmonic_energy) / np.array(fundamental_energy))
    else:
        harm_to_fund_ratio = 0


    # 1.1 Total Harmonic Distortion
    # -----------------------------
    thd = calculate_thd(original, distorted, sample_rate)
    




    
    # 2. Crest Factor Reduction
    # -------------------------
    # Measure how much the signal has been compressed/limited
    orig_crest = np.max(np.abs(original)) / np.sqrt(np.mean(original**2))
    dist_crest = np.max(np.abs(distorted)) / np.sqrt(np.mean(distorted**2))
    crest_factor_ratio = orig_crest / dist_crest  # Higher = more compression
    
    # 3. Waveform Asymmetry
    # ---------------------
    # Distortion often causes asymmetric waveforms
    orig_asymmetry = np.mean(np.abs(original)) / np.sqrt(np.mean(original**2))
    dist_asymmetry = np.mean(np.abs(distorted)) / np.sqrt(np.mean(distorted**2))
    asymmetry_change = dist_asymmetry / orig_asymmetry
    
    # 4. Zero-crossing Rate Change
    # ---------------------------
    # Distortion adds harmonics which increases zero-crossing rate
    orig_zcr = np.mean(librosa.feature.zero_crossing_rate(original))
    dist_zcr = np.mean(librosa.feature.zero_crossing_rate(distorted))
    zcr_ratio = dist_zcr / orig_zcr
    
    # 5. Clipping Detection
    # --------------------
    # Measure the amount of hard clipping
    clip_threshold = 0.95
    orig_clip_percent = np.sum(np.abs(original) > clip_threshold) / len(original)
    dist_clip_percent = np.sum(np.abs(distorted) > clip_threshold) / len(distorted)
    clipping_increase = dist_clip_percent - orig_clip_percent
    
    # 6. Nonlinear Distortion Index (perceptually weighted)
    # ---------------------------------------------
    # Perceptually-weighted difference in spectra
    
    # Get the perceptually weighted spectrogram (A-weighting)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(freqs)
    
    # Apply A-weighting to both spectrograms
    orig_weighted = orig_mag * np.reshape(10**(a_weighting/20), (-1, 1))
    dist_weighted = dist_mag * np.reshape(10**(a_weighting/20), (-1, 1))
    
    # Calculate perceptual difference
    perceptual_diff = np.mean((dist_weighted - orig_weighted)**2)


    # Spectral centroid of the distorted 
    # ---------------------------------
    # Higher spectral centroid indicates more high-frequency content
    speccentroid = np.mean(librosa.feature.spectral_centroid(S=dist_mag, sr=sample_rate))

    
    return {
        'harmonic_ratio': harm_to_fund_ratio,
        'crest_factor_ratio': crest_factor_ratio,
        'asymmetry_change': asymmetry_change,
        'zcr_ratio': zcr_ratio,
        'clipping_increase': clipping_increase,
        'perceptual_diff': perceptual_diff,
        'thd': thd,
        'spectral_centroid': speccentroid,
    }


def plot_waveforms(clean_signal, argminmeanmax_perceptual):
    # Visualize waveforms and spectra for selected settings
    plt.figure(figsize=(20, 12))
    plt.suptitle(f"Distortion Analysis for {PEDAL}")
    
    # Select a small portion for waveform analysis (0.01 sec)
    start_idx = len(clean_signal) // 2
    window_size = int(0.01 * SR)

    assert type(argminmeanmax_perceptual) is tuple and len(argminmeanmax_perceptual) == 3, "argminmeanmax_perceptual must be a tuple of 3 values"
    
    argmin_perceptual,argmean_perceptual,argmax_perceptual = argminmeanmax_perceptual

    # Waveform comparison - low distortion
    plt.subplot(3, 3, 1)
    low_y, low_x = argmin_perceptual
    low_dist = get_wetsweep(gain=low_y, tone=low_x)
    plt.plot(clean_signal[start_idx:start_idx+window_size], label="Original")
    plt.plot(low_dist[start_idx:start_idx+window_size], label="Processed", alpha=0.7)
    plt.title(f"Low Distortion Waveform (gain={low_y}, tone={low_x})")
    plt.legend()
    
    # Waveform comparison - medium distortion
    plt.subplot(3, 3, 2)
    med_y, med_x = argmean_perceptual
    med_dist = get_wetsweep(gain=med_y, tone=med_x)
    plt.plot(clean_signal[start_idx:start_idx+window_size], label="Original")
    plt.plot(med_dist[start_idx:start_idx+window_size], label="Processed", alpha=0.7)
    plt.title(f"Medium Distortion Waveform (gain={med_y}, tone={med_x})")
    plt.legend()
    
    # Waveform comparison - high distortion
    plt.subplot(3, 3, 3)
    high_y, high_x = argmax_perceptual
    high_dist = get_wetsweep(gain=high_y, tone=high_x)
    plt.plot(clean_signal[start_idx:start_idx+window_size], label="Original")
    plt.plot(high_dist[start_idx:start_idx+window_size], label="Processed", alpha=0.7)
    plt.title(f"High Distortion Waveform (gain={high_y}, tone={high_x})")
    plt.legend()
    
    # Spectrum comparison - low distortion
    plt.subplot(3, 3, 4)
    orig_spec = np.abs(np.fft.rfft(clean_signal[start_idx:start_idx+window_size*10]))
    low_spec = np.abs(np.fft.rfft(low_dist[start_idx:start_idx+window_size*10]))
    freq = np.fft.rfftfreq(window_size*10, 1/SR)
    plt.semilogx(freq, 20 * np.log10(orig_spec + 1e-9), label="Original")
    plt.semilogx(freq, 20 * np.log10(low_spec + 1e-9), label="Processed", alpha=0.7)
    plt.title("Low Distortion Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlim(20, 20000)
    
    # Spectrum comparison - medium distortion
    plt.subplot(3, 3, 5)
    med_spec = np.abs(np.fft.rfft(med_dist[start_idx:start_idx+window_size*10]))
    plt.semilogx(freq, 20 * np.log10(orig_spec + 1e-9), label="Original")
    plt.semilogx(freq, 20 * np.log10(med_spec + 1e-9), label="Processed", alpha=0.7)
    plt.title("Medium Distortion Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlim(20, 20000)
    
    # Spectrum comparison - high distortion
    plt.subplot(3, 3, 6)
    high_spec = np.abs(np.fft.rfft(high_dist[start_idx:start_idx+window_size*10]))
    plt.semilogx(freq, 20 * np.log10(orig_spec + 1e-9), label="Original")
    plt.semilogx(freq, 20 * np.log10(high_spec + 1e-9), label="Processed", alpha=0.7)
    plt.title("High Distortion Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlim(20, 20000)
    
    # Ratio between spectra - low distortion
    plt.subplot(3, 3, 7)
    ratio_low = low_spec / (orig_spec + 1e-9)
    plt.semilogx(freq, ratio_low)
    plt.title("Low Distortion Spectral Ratio")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Ratio")
    plt.grid(True, which="both", ls="--")
    plt.xlim(20, 20000)
    
    # Ratio between spectra - medium distortion
    plt.subplot(3, 3, 8)
    ratio_med = med_spec / (orig_spec + 1e-9)
    plt.semilogx(freq, ratio_med)
    plt.title("Medium Distortion Spectral Ratio")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Ratio")
    plt.grid(True, which="both", ls="--")
    plt.xlim(20, 20000)
    
    # Ratio between spectra - high distortion
    plt.subplot(3, 3, 9)
    ratio_high = high_spec / (orig_spec + 1e-9)
    plt.semilogx(freq, ratio_high)
    plt.title("High Distortion Spectral Ratio")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Ratio")
    plt.grid(True, which="both", ls="--")
    plt.xlim(20, 20000)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{PEDAL}_distortion_improved_waveform_spectrum_analysis.png"))
    plt.show()


def plot_heatmaps(results_harmonic, results_crest, results_perceptual, tone_range, gain_range, suptitle = "Distortion Analysis", axlabels = ['X','Y'], savepath="dist.png",annotations=False, results_thd = None, results_speccentroid = None, setting_points=None):

    # Create heatmaps for each metric
    

    # Harmonic Ratio Heatmap
    # plt.subplot(1, plots, 1)
    # plt.suptitle(suptitle)
    # sns.heatmap(results_harmonic, 
    #             xticklabels=[f"{t:.1f}" for t in tone_range],
    #             yticklabels=[f"{g:.1f}" for g in gain_range],
    #             annot=annotations, fmt=".1f", cmap="coolwarm")
    # plt.xlabel(axlabels[0])
    # plt.ylabel(axlabels[1])
    # plt.title("Harmonic-to-Fundamental Ratio")
    # plt.gca().invert_yaxis()
    
    # Crest Factor Ratio Heatmap
    plt.figure(figsize=(5,4.2))
    sns.heatmap(results_crest, 
                xticklabels=[f"{t:.1f}" for t in tone_range],
                yticklabels=[f"{g:.1f}" for g in gain_range],
                annot=annotations, fmt=".2f", cmap="coolwarm")
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1])
    plt.yticks([])
    plt.xticks([])
    plt.title("Crest Factor Ratio (Compression)")
    plt.gca().invert_yaxis()

    # Plot white dots from the setting points over the heatmap
    if setting_points is not None:
        setting_points_xy =  setting_points.loc[:,['x','y','label_name','label_name']].values
        # Replace the label name with a marker from list ['x','o','s','^','v','<','>','d','p','P','*','h','H','+','X','D','|','_']
        markers = ['x','2','+','o','v','<','>','d','p','P','*','h','H','+','X','D','|','_']
        unique_labels = setting_points['label_name'].unique()
        label_to_marker = dict(zip(unique_labels, markers))
        for i in range(len(setting_points_xy)):
            setting_points_xy[i][2] = label_to_marker[setting_points_xy[i][2]]


        SCALE = len(tone_range)
        for x, y, marker, label in setting_points_xy:
            plt.scatter(x*SCALE, y*SCALE, 
                        facecolors='w' if marker != 'o' else 'None',
                        edgecolors= 'w' if marker == 'o' else 'None', 
                        marker=marker, s=45,
                        label=label)



    plt.tight_layout()
    plt.savefig(savepath.replace('.png','_crest.png'), bbox_inches='tight')
    
    # Perceptual Difference Heatmap
    # plt.subplot(1, plots, 2)
    # sns.heatmap(results_perceptual, 
    #             xticklabels=[f"{t:.1f}" for t in tone_range],
    #             yticklabels=[f"{g:.1f}" for g in gain_range],
    #             annot=annotations, fmt=".1f", cmap="coolwarm")
    # plt.xlabel(axlabels[0])
    # plt.ylabel(axlabels[1])
    # plt.title("Perceptual Distortion (A-weighted)")
    # plt.gca().invert_yaxis()

    from matplotlib.colors import LogNorm

    if results_thd is not None:
        
        plt.figure(figsize=(5,4.2))
        # sns.heatmap(results_thd, 
        #             xticklabels=[f"{t:.1f}" for t in tone_range],
        #             yticklabels=[f"{g:.1f}" for g in gain_range],
        #             annot=annotations, fmt=".1f", cmap="coolwarm")

        # heatmap with log scale for THD
        vmax = np.max(results_thd)
        vmin = np.min(results_thd)
        sns.heatmap(results_thd,
                    xticklabels=[f"{t:.1f}" for t in tone_range],
                    yticklabels=[f"{g:.1f}" for g in gain_range],
                    annot=annotations, fmt=".2f", cmap="magma", norm=LogNorm(vmin=vmin, vmax= vmax))
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
        plt.yticks([])
        plt.xticks([])
        plt.title("Total Harmonic Distortion")
        plt.gca().invert_yaxis()

        if setting_points is not None:
            for x, y, marker, label in setting_points_xy:
                plt.scatter(x*SCALE, y*SCALE, facecolors='w' if marker != 'o' else 'none', marker=marker, s=45, edgecolors= 'w' if marker == 'o' else 'none',
                            label=label)
        plt.tight_layout()
        plt.savefig(savepath.replace('.png','_thd.png'), bbox_inches='tight')

    if results_speccentroid is not None:
        
        plt.figure(figsize=(5,4.2))
        sns.heatmap(results_speccentroid, 
                    xticklabels=[f"{t:.1f}" for t in tone_range],
                    yticklabels=[f"{g:.1f}" for g in gain_range],
                    annot=annotations, fmt=".1f", cmap=sns.cubehelix_palette(as_cmap=True, reverse=True))
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
        plt.yticks([])
        plt.xticks([])
        plt.title("Spectral Centroid")
        plt.gca().invert_yaxis()

        if setting_points is not None:
            for x, y, marker, label in setting_points_xy:
                plt.scatter(x*SCALE, y*SCALE, facecolors='w' if marker != 'o' else 'none', marker=marker, s=45, edgecolors= 'w' if marker == 'o' else 'none',
                            label=label)

        plt.tight_layout()
        plt.savefig(savepath.replace('.png','_speccentroid.png'), bbox_inches='tight')

    
    # # Save only the legend with the scatter markers (in black though) to a separate image file
    # legendfig, ax = plt.subplots()  

    # labels = [v[3] for v in setting_points_xy]
    # uniquemarkers = list(set([label_to_marker[label] for label in labels]))

    # for marker in uniquemarkers:
    #     xs = [v[0] for v in setting_points_xy if v[2] == marker]
    #     ys = [v[1] for v in setting_points_xy if v[2] == marker]
    #     label = [v[3] for v in setting_points_xy if v[2] == marker][0]
    #     ax.scatter(xs*SCALE, ys*SCALE, facecolors='k', marker=marker, s=45, edgecolors= 'k', label=label)
    # ax.legend()
    # ax.axis('off')
    # legendfig.savefig(savepath.replace('.png','_speccentroid_legend.png'), bbox_inches='tight')
    # print(f"Saved legend to {savepath.replace('.png','_speccentroid_legend.png')}")

    import matplotlib.patches as mpatches

    # Create a new figure for the legend
    legendfig = plt.figure(figsize=(5, 5))  # Adjust size as needed

    # Create handles for the legend manually
    handles = []
    labels = []

    # Get unique markers and their corresponding labels
    unique_markers_and_labels = set([(label_to_marker[label], label) for label in set([v[3] for v in setting_points_xy])])

    unique_markers_and_labels = [(v[0], datasetMetadataRenamer.datasetname2shortname(v[1])) for v in unique_markers_and_labels]

    unique_labels = [v[1] for v in unique_markers_and_labels]
    if sorted(unique_labels) == sorted(['HON', 'ZEN', 'FEL', 'SOU']):
        # order unique_markers_and_labels according to ['HON', 'ZEN', 'FEL', 'SOU']
        unique_markers_and_labels = sorted(unique_markers_and_labels, key=lambda x: ['HON', 'ZEN', 'FEL', 'SOU'].index(x[1]))

    # Create handles for each marker-label pair
    for marker, label in unique_markers_and_labels:
        # Create a handle with the marker
        handle = plt.scatter([], [], marker=marker, s=45, 
                            facecolors='k' if marker != 'o' else 'w',
                            edgecolors= 'k' if marker == 'o' else 'w', 
                            label=label)
        handles.append(handle)
        labels.append(label)

    print('labels',labels)

    # Create the legend
    # legend = plt.legend(handles=handles, labels=labels, loc='center', frameon=False, prop={'size': 16})
    # one row, all horizontal
    # legend = plt.legend(handles=handles, labels=labels, loc='center', ncol=1, title='Pedal', )

    # One column, vertical, with added spacing 
    legend = plt.legend(handles=handles, labels=labels, loc='center', ncol=1, title='Pedal', 
                        borderpad=1, labelspacing=1, handlelength=0.5, handletextpad=0.5, columnspacing=1)

    # Remove everything else from the figure
    ax = plt.gca()
    ax.set_axis_off()

    # Save only the legend
    legendfig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(legendfig.dpi_scale_trans.inverted())
    savename = os.path.join(os.path.dirname(savepath), f"legend.pdf")
    legendfig.savefig(savename, 
                    bbox_inches=legend_bbox, 
                    pad_inches=0.1)

    plt.close(legendfig)
    print(f"Saved legend to {savename}")



def evaluate_distortion_grid():
    """
    Evaluate distortion using improved metrics across a grid of gain and tone parameters.
    """
    # Load dry sweep
    clean_signal = get_drysweep()
    
    # Define parameter ranges
    gain_range = np.linspace(0, 5, 6)
    tone_range = np.linspace(0, 5, 6)
    
    # Initialize results matrices
    results_harmonic = np.zeros((len(gain_range), len(tone_range)))
    results_crest = np.zeros((len(gain_range), len(tone_range)))
    results_perceptual = np.zeros((len(gain_range), len(tone_range)))
    results_speccentroid = np.zeros((len(gain_range), len(tone_range)))
    
    
    # Perform grid search
    for i, gain in enumerate(gain_range):
        for j, tone in enumerate(tone_range):
            # Get distorted signal
            distorted = get_wetsweep(gain=gain, tone=tone)
            
            # Calculate improved metrics
            metrics = calculate_improved_distortion_metrics(clean_signal, distorted)
            
            # Store results
            results_harmonic[i, j] = metrics['harmonic_ratio']
            results_crest[i, j] = metrics['crest_factor_ratio']
            results_perceptual[i, j] = metrics['perceptual_diff']
            results_speccentroid[i, j] = metrics['spectral_centroid']

    argmax_perceptual = np.unravel_index(np.argmax(results_perceptual), results_perceptual.shape)
    argmin_perceptual = np.unravel_index(np.argmin(results_perceptual), results_perceptual.shape)
    argmean_perceptual = np.unravel_index(np.argmin(np.abs(results_perceptual - np.mean(results_perceptual))), results_perceptual.shape)
    

    plot_heatmaps(results_harmonic, results_crest, results_perceptual, tone_range, gain_range,
                  f"Distortion Analysis for {PEDAL}",
                  ['Tone', 'Gain'],
                  savepath=os.path.join(OUTDIR, f"{PEDAL}_distortion_improved_metrics_heatmap.png"),
                  results_speccentroid = results_speccentroid)
    
    
    plot_waveforms(clean_signal, (argmin_perceptual,argmean_perceptual,argmax_perceptual))
    
    return {
        'harmonic_ratio': results_harmonic,
        'crest_factor': results_crest, 
        'perceptual_diff': results_perceptual,
    }


# Run the analysis
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change working directory to the script directory

    SR = 48000
    SWEEP_NAME = 's0'
    PEDAL = "bigfella"
    DRY_AUDIO_PATH = f"../network/dataset/input/{SWEEP_NAME}_0-unprocessed_input.wav"
    assert os.path.exists(DRY_AUDIO_PATH), f"Input audio file not found at {DRY_AUDIO_PATH}"
    WET_AUDIO_FOLDER = f"../network/dataset/{PEDAL}/"
    assert os.path.exists(WET_AUDIO_FOLDER), f"Pedal audio folder not found at {WET_AUDIO_FOLDER}"

    OUTDIR = 'outplots'
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
        
    results = evaluate_distortion_grid()