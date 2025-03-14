import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import librosa
import os
import soundfile as sf
from scipy.signal import find_peaks

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

WAVEFORM_COMBINED_MINMAX = False


def get_drysweep(sample_rate=SR):
    """Load the dry sweep from file"""
    sweep, _ = librosa.load(DRY_AUDIO_PATH, sr=sample_rate)
    return sweep


def get_wetsweep(gain=1.0, tone=3.0):
    """Load the wet sweep from file"""
    filepath = os.path.join(WET_AUDIO_FOLDER, f"{SWEEP_NAME}_{PEDAL}_g{int(gain)}_t{int(tone)}.wav")
    assert os.path.exists(filepath), f"Pedal audio file not found at {filepath}"
    audio, _ = librosa.load(filepath, sr=SR)
    return audio


def calculate_improved_distortion_metrics(original, distorted):
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
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    a_weighting = librosa.A_weighting(freqs)
    
    # Apply A-weighting to both spectrograms
    orig_weighted = orig_mag * np.reshape(10**(a_weighting/20), (-1, 1))
    dist_weighted = dist_mag * np.reshape(10**(a_weighting/20), (-1, 1))
    
    # Calculate perceptual difference
    perceptual_diff = np.mean((dist_weighted - orig_weighted)**2)
    
    return {
        'harmonic_ratio': harm_to_fund_ratio,
        'crest_factor_ratio': crest_factor_ratio,
        'asymmetry_change': asymmetry_change,
        'zcr_ratio': zcr_ratio,
        'clipping_increase': clipping_increase,
        'perceptual_diff': perceptual_diff
    }


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

    argmax_perceptual = np.unravel_index(np.argmax(results_perceptual), results_perceptual.shape)
    argmin_perceptual = np.unravel_index(np.argmin(results_perceptual), results_perceptual.shape)
    argmean_perceptual = np.unravel_index(np.argmin(np.abs(results_perceptual - np.mean(results_perceptual))), results_perceptual.shape)
    
    # Create heatmaps for each metric
    plt.figure(figsize=(18, 6))
    
    # Harmonic Ratio Heatmap
    plt.subplot(1, 3, 1)
    plt.suptitle(f"Distortion Analysis for {PEDAL}")
    sns.heatmap(results_harmonic, 
                xticklabels=[f"{t:.1f}" for t in tone_range],
                yticklabels=[f"{g:.1f}" for g in gain_range],
                annot=True, fmt=".1f", cmap="viridis")
    plt.xlabel("Tone Parameter")
    plt.ylabel("Gain Parameter")
    plt.title("Harmonic-to-Fundamental Ratio")
    plt.gca().invert_yaxis()
    
    # Crest Factor Ratio Heatmap
    plt.subplot(1, 3, 2)
    sns.heatmap(results_crest, 
                xticklabels=[f"{t:.1f}" for t in tone_range],
                yticklabels=[f"{g:.1f}" for g in gain_range],
                annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("Tone Parameter")
    plt.ylabel("Gain Parameter")
    plt.title("Crest Factor Ratio (Compression)")
    plt.gca().invert_yaxis()
    
    # Perceptual Difference Heatmap
    plt.subplot(1, 3, 3)
    sns.heatmap(results_perceptual, 
                xticklabels=[f"{t:.1f}" for t in tone_range],
                yticklabels=[f"{g:.1f}" for g in gain_range],
                annot=True, fmt=".1f", cmap="viridis")
    plt.xlabel("Tone Parameter")
    plt.ylabel("Gain Parameter")
    plt.title("Perceptual Distortion (A-weighted)")
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{PEDAL}_distortion_improved_metrics_heatmap.png"))
    plt.show()
    
    # Create a combined distortion metric
    combined_metric = (
        results_harmonic / np.max(results_harmonic) * 0.5 + 
        results_crest / np.max(results_crest) * 0.3 + 
        results_perceptual / np.max(results_perceptual) * 0.2
    )

    argmin_combined = np.unravel_index(np.argmin(combined_metric), combined_metric.shape)
    argmax_combined = np.unravel_index(np.argmax(combined_metric), combined_metric.shape)
    argmean_combined = np.unravel_index(np.argmin(np.abs(combined_metric - np.mean(combined_metric))), combined_metric.shape)
    
    # plt.figure(figsize=(10, 8))
    # plt.suptitle(f"Distortion Analysis for {PEDAL}")

    # sns.heatmap(combined_metric, 
    #             xticklabels=[f"{t:.1f}" for t in tone_range],
    #             yticklabels=[f"{g:.1f}" for g in gain_range],
    #             annot=True, fmt=".1f", cmap="viridis")
    # plt.xlabel("Tone Parameter")
    # plt.ylabel("Gain Parameter")
    # plt.title("Combined Distortion Metric")
    # plt.gca().invert_yaxis()
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTDIR, f"{PEDAL}_distortion_improved_combined_distortion_metric.png"))
    # plt.show()
    
    # Visualize waveforms and spectra for selected settings
    plt.figure(figsize=(20, 12))
    plt.suptitle(f"Distortion Analysis for {PEDAL}")
    
    # Select a small portion for waveform analysis (0.01 sec)
    start_idx = len(clean_signal) // 2
    window_size = int(0.01 * SR)


    # print('argmin_perceptual', argmin_perceptual)
    # print('argmax_perceptual', argmax_perceptual)
    # print('argmean_perceptual', argmean_perceptual)
    # exit()
    
    # Waveform comparison - low distortion
    plt.subplot(3, 3, 1)
    low_gain, low_tone = argmin_perceptual if not WAVEFORM_COMBINED_MINMAX else argmin_perceptual
    low_dist = get_wetsweep(gain=low_gain, tone=low_tone)
    plt.plot(clean_signal[start_idx:start_idx+window_size], label="Original")
    plt.plot(low_dist[start_idx:start_idx+window_size], label="Processed", alpha=0.7)
    plt.title(f"Low Distortion Waveform (gain={low_gain}, tone={low_tone})")
    plt.legend()
    
    # Waveform comparison - medium distortion
    plt.subplot(3, 3, 2)
    med_gain, med_tone = argmean_perceptual if not WAVEFORM_COMBINED_MINMAX else argmean_combined
    med_dist = get_wetsweep(gain=med_gain, tone=med_tone)
    plt.plot(clean_signal[start_idx:start_idx+window_size], label="Original")
    plt.plot(med_dist[start_idx:start_idx+window_size], label="Processed", alpha=0.7)
    plt.title(f"Medium Distortion Waveform (gain={med_gain}, tone={med_tone})")
    plt.legend()
    
    # Waveform comparison - high distortion
    plt.subplot(3, 3, 3)
    high_gain, high_tone = argmax_perceptual if not WAVEFORM_COMBINED_MINMAX else argmax_combined
    high_dist = get_wetsweep(gain=high_gain, tone=high_tone)
    plt.plot(clean_signal[start_idx:start_idx+window_size], label="Original")
    plt.plot(high_dist[start_idx:start_idx+window_size], label="Processed", alpha=0.7)
    plt.title(f"High Distortion Waveform (gain={high_gain}, tone={high_tone})")
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
    
    return {
        'harmonic_ratio': results_harmonic,
        'crest_factor': results_crest, 
        'perceptual_diff': results_perceptual,
        'combined': combined_metric
    }


# Run the analysis
if __name__ == "__main__":
    results = evaluate_distortion_grid()