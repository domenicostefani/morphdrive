import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import os
import soundfile as sf
from scipy import signal
from scipy.signal import find_peaks, butter, sosfreqz, sosfilt

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


def calculate_filter_metrics(original, filtered):
    """
    Calculate metrics that characterize filtering effects
    """
    # Ensure signals are the same length
    min_len = min(len(original), len(filtered))
    original = original[:min_len]
    filtered = filtered[:min_len]
    
    # Normalize levels for fair comparison
    original = original / np.max(np.abs(original))
    filtered = filtered / np.max(np.abs(filtered))
    
    # 1. Spectral Centroid
    # --------------------
    # Measure of the "center of mass" of the spectrum
    orig_centroid = np.mean(librosa.feature.spectral_centroid(y=original, sr=SR))
    filt_centroid = np.mean(librosa.feature.spectral_centroid(y=filtered, sr=SR))
    centroid_shift = filt_centroid / orig_centroid  # < 1 for lowpass, > 1 for highpass
    
    # 2. Filter Cutoff Estimation
    # --------------------------
    # Estimate filter cutoff by analyzing transfer function
    n_fft = 2048
    # Calculate magnitude spectra
    orig_spec = np.abs(librosa.stft(original, n_fft=n_fft))
    filt_spec = np.abs(librosa.stft(filtered, n_fft=n_fft))
    
    # Calculate transfer function (frequency response)
    transfer_func = np.mean(filt_spec / (orig_spec + 1e-10), axis=1)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    
    # Find -3dB points (cutoff frequencies)
    transfer_func_db = 20 * np.log10(transfer_func + 1e-10)
    max_gain = np.max(transfer_func_db)
    cutoff_level = max_gain - 3
    
    # Find low cutoff (for highpass or bandpass)
    low_cutoff = None
    for i, level in enumerate(transfer_func_db):
        if level >= cutoff_level and i > 0:
            # Interpolate to find more precise cutoff
            prev_freq = freqs[i-1]
            prev_level = transfer_func_db[i-1]
            curr_freq = freqs[i]
            curr_level = level
            
            if prev_level < cutoff_level < curr_level:
                ratio = (cutoff_level - prev_level) / (curr_level - prev_level)
                low_cutoff = prev_freq + ratio * (curr_freq - prev_freq)
                break
    
    # Find high cutoff (for lowpass or bandpass)
    high_cutoff = None
    for i in range(len(transfer_func_db)-1, 0, -1):
        if transfer_func_db[i] >= cutoff_level and i < len(transfer_func_db)-1:
            # Interpolate to find more precise cutoff
            next_freq = freqs[i+1]
            next_level = transfer_func_db[i+1]
            curr_freq = freqs[i]
            curr_level = transfer_func_db[i]
            
            if curr_level > cutoff_level > next_level:
                ratio = (cutoff_level - next_level) / (curr_level - next_level)
                high_cutoff = curr_freq + ratio * (next_freq - curr_freq)
                break
    
    # Set default values if cutoffs weren't found
    if low_cutoff is None:
        low_cutoff = 20  # Assume 20 Hz if no low cutoff found
    if high_cutoff is None:
        high_cutoff = SR/2  # Assume Nyquist if no high cutoff found
    
    # 3. Filter Q-factor (sharpness of resonance)
    # ------------------------------------------
    # Estimate Q-factor from bandwidth relative to center frequency
    if low_cutoff != 20 and high_cutoff != SR/2:
        # It's a bandpass
        center_freq = np.sqrt(low_cutoff * high_cutoff)
        bandwidth = high_cutoff - low_cutoff
        q_factor = center_freq / bandwidth
    elif low_cutoff == 20:
        # It's a lowpass
        q_factor = high_cutoff / (high_cutoff * 0.707)  # Approximate Q at cutoff
    else:
        # It's a highpass
        q_factor = low_cutoff / (low_cutoff * 0.707)  # Approximate Q at cutoff
    
    # 4. Spectral Tilt / Slope
    # -----------------------
    # Compute the overall spectral tilt (slope of spectrum in log-log space)
    freqs_log = np.log10(freqs[1:])  # Exclude DC
    orig_spec_avg = np.mean(orig_spec[1:, :], axis=1)  # Exclude DC
    filt_spec_avg = np.mean(filt_spec[1:, :], axis=1)  # Exclude DC
    
    # Linear regression to find slope
    orig_spec_log = np.log10(orig_spec_avg + 1e-10)
    filt_spec_log = np.log10(filt_spec_avg + 1e-10)
    
    # Calculate slope for original
    A_orig = np.vstack([freqs_log, np.ones(len(freqs_log))]).T
    slope_orig, _ = np.linalg.lstsq(A_orig, orig_spec_log, rcond=None)[0]
    
    # Calculate slope for filtered
    A_filt = np.vstack([freqs_log, np.ones(len(freqs_log))]).T
    slope_filt, _ = np.linalg.lstsq(A_filt, filt_spec_log, rcond=None)[0]
    
    # Relative change in spectral slope
    spectral_tilt = slope_filt - slope_orig
    
    # 5. Spectral Rolloff
    # ------------------
    # Frequency below which X% of spectral energy resides
    orig_rolloff = np.mean(librosa.feature.spectral_rolloff(y=original, sr=SR, roll_percent=0.85))
    filt_rolloff = np.mean(librosa.feature.spectral_rolloff(y=filtered, sr=SR, roll_percent=0.85))
    rolloff_ratio = filt_rolloff / orig_rolloff
    
    return {
        'centroid_shift': centroid_shift,
        'low_cutoff': low_cutoff,
        'high_cutoff': high_cutoff,
        'q_factor': q_factor,
        'spectral_tilt': spectral_tilt,
        'rolloff_ratio': rolloff_ratio,
    }


def evaluate_filter_grid():
    """
    Evaluate filtering characteristics across a grid of gain and tone parameters.
    """
    # Load dry sweep
    clean_signal = get_drysweep()
    
    # Define parameter ranges
    gain_range = np.linspace(0, 5, 6)
    tone_range = np.linspace(0, 5, 6)
    
    # Initialize results matrices
    results_centroid = np.zeros((len(gain_range), len(tone_range)))
    results_low_cutoff = np.zeros((len(gain_range), len(tone_range)))
    results_high_cutoff = np.zeros((len(gain_range), len(tone_range)))
    results_q_factor = np.zeros((len(gain_range), len(tone_range)))
    results_tilt = np.zeros((len(gain_range), len(tone_range)))
    
    # Perform grid search
    for i, gain in enumerate(gain_range):
        for j, tone in enumerate(tone_range):
            # Get filtered signal
            filtered = get_wetsweep(gain=gain, tone=tone)
            
            # Calculate filter metrics
            metrics = calculate_filter_metrics(clean_signal, filtered)
            
            # Store results
            results_centroid[i, j] = metrics['centroid_shift']
            results_low_cutoff[i, j] = metrics['low_cutoff']
            results_high_cutoff[i, j] = metrics['high_cutoff']
            results_q_factor[i, j] = metrics['q_factor']
            results_tilt[i, j] = metrics['spectral_tilt']
    
    # Create heatmaps for key metrics
    plt.figure(figsize=(18, 12))
    plt.suptitle(f"Filter Analysis for {PEDAL}")
    
    # Spectral Centroid Shift Heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(results_centroid, 
                xticklabels=[f"{t:.1f}" for t in tone_range],
                yticklabels=[f"{g:.1f}" for g in gain_range],
                annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("Tone Parameter")
    plt.ylabel("Gain Parameter")
    plt.title("Spectral Centroid Shift\n(>1 = brighter, <1 = darker)")
    plt.gca().invert_yaxis()
    
    # Low Cutoff Frequency Heatmap
    plt.subplot(2, 2, 2)
    sns.heatmap(results_low_cutoff, 
                xticklabels=[f"{t:.1f}" for t in tone_range],
                yticklabels=[f"{g:.1f}" for g in gain_range],
                annot=True, fmt=".0f", cmap="viridis")
    plt.xlabel("Tone Parameter")
    plt.ylabel("Gain Parameter")
    plt.title("Low Cutoff Frequency (Hz)")
    plt.gca().invert_yaxis()
    
    # High Cutoff Frequency Heatmap
    plt.subplot(2, 2, 3)
    sns.heatmap(results_high_cutoff / 1000, 
                xticklabels=[f"{t:.1f}" for t in tone_range],
                yticklabels=[f"{g:.1f}" for g in gain_range],
                annot=True, fmt=".1f", cmap="viridis")
    plt.xlabel("Tone Parameter")
    plt.ylabel("Gain Parameter")
    plt.title("High Cutoff Frequency (kHz)")
    plt.gca().invert_yaxis()
    
    # Filter Q-Factor Heatmap
    plt.subplot(2, 2, 4)
    sns.heatmap(results_q_factor, 
                xticklabels=[f"{t:.1f}" for t in tone_range],
                yticklabels=[f"{g:.1f}" for g in gain_range],
                annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("Tone Parameter")
    plt.ylabel("Gain Parameter")
    plt.title("Filter Q-Factor (Resonance)")
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{PEDAL}_filter_metrics_heatmap.png"))
    plt.show()
    
    # Visualize frequency response for selected filter settings
    plt.figure(figsize=(18, 10))
    plt.suptitle(f"Frequency Response Analysis for {PEDAL}")
    
    # Select interesting points from the grid
    min_centroid_idx = np.unravel_index(np.argmin(results_centroid), results_centroid.shape)
    max_centroid_idx = np.unravel_index(np.argmax(results_centroid), results_centroid.shape)
    max_q_idx = np.unravel_index(np.argmax(results_q_factor), results_q_factor.shape)
    
    min_centroid_gain, min_centroid_tone = gain_range[min_centroid_idx[0]], tone_range[min_centroid_idx[1]]
    max_centroid_gain, max_centroid_tone = gain_range[max_centroid_idx[0]], tone_range[max_centroid_idx[1]]
    max_q_gain, max_q_tone = gain_range[max_q_idx[0]], tone_range[max_q_idx[1]]
    
    # Get signals
    min_centroid_signal = get_wetsweep(gain=min_centroid_gain, tone=min_centroid_tone)
    max_centroid_signal = get_wetsweep(gain=max_centroid_gain, tone=max_centroid_tone)
    max_q_signal = get_wetsweep(gain=max_q_gain, tone=max_q_tone)
    
    # Calculate frequency responses
    n_fft = 4096
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    
    orig_spec = np.abs(librosa.stft(clean_signal, n_fft=n_fft))
    min_centroid_spec = np.abs(librosa.stft(min_centroid_signal, n_fft=n_fft))
    max_centroid_spec = np.abs(librosa.stft(max_centroid_signal, n_fft=n_fft))
    max_q_spec = np.abs(librosa.stft(max_q_signal, n_fft=n_fft))
    
    # Average over time
    orig_spec_avg = np.mean(orig_spec, axis=1)
    min_centroid_spec_avg = np.mean(min_centroid_spec, axis=1)
    max_centroid_spec_avg = np.mean(max_centroid_spec, axis=1)
    max_q_spec_avg = np.mean(max_q_spec, axis=1)
    
    # Calculate transfer functions
    min_centroid_tf = min_centroid_spec_avg / (orig_spec_avg + 1e-10)
    max_centroid_tf = max_centroid_spec_avg / (orig_spec_avg + 1e-10)
    max_q_tf = max_q_spec_avg / (orig_spec_avg + 1e-10)
    
    # Plot frequency responses
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs, 20 * np.log10(min_centroid_tf + 1e-10), label=f"Darkest Filter (gain={min_centroid_gain}, tone={min_centroid_tone})")
    plt.semilogx(freqs, 20 * np.log10(max_centroid_tf + 1e-10), label=f"Brightest Filter (gain={max_centroid_gain}, tone={max_centroid_tone})")
    plt.semilogx(freqs, 20 * np.log10(max_q_tf + 1e-10), label=f"Most Resonant (gain={max_q_gain}, tone={max_q_tone})")
    plt.title("Frequency Response Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.xlim(20, 20000)
    
    # Normalized frequency responses (for shape comparison)
    plt.subplot(2, 1, 2)
    min_centroid_tf_norm = min_centroid_tf / np.max(min_centroid_tf)
    max_centroid_tf_norm = max_centroid_tf / np.max(max_centroid_tf)
    max_q_tf_norm = max_q_tf / np.max(max_q_tf)
    
    plt.semilogx(freqs, 20 * np.log10(min_centroid_tf_norm + 1e-10), label=f"Darkest Filter (gain={min_centroid_gain}, tone={min_centroid_tone})")
    plt.semilogx(freqs, 20 * np.log10(max_centroid_tf_norm + 1e-10), label=f"Brightest Filter (gain={max_centroid_gain}, tone={max_centroid_tone})")
    plt.semilogx(freqs, 20 * np.log10(max_q_tf_norm + 1e-10), label=f"Most Resonant (gain={max_q_gain}, tone={max_q_tone})")
    plt.title("Normalized Frequency Response Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude (dB)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.xlim(20, 20000)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{PEDAL}_filter_frequency_response.png"))
    plt.show()
    
    # Visualize filter type classification
    plt.figure(figsize=(10, 8))
    
    # Create filter type classification based on cutoffs
    filter_types = np.zeros(results_centroid.shape, dtype=int)
    # 0: Broadband, 1: Lowpass, 2: Highpass, 3: Bandpass
    
    for i in range(len(gain_range)):
        for j in range(len(tone_range)):
            low_cut = results_low_cutoff[i, j]
            high_cut = results_high_cutoff[i, j]
            
            if low_cut <= 30 and high_cut >= SR/2 - 1000:
                filter_types[i, j] = 0  # Broadband (no significant filtering)
            elif low_cut <= 30 and high_cut < SR/2 - 1000:
                filter_types[i, j] = 1  # Lowpass
            elif low_cut > 30 and high_cut >= SR/2 - 1000:
                filter_types[i, j] = 2  # Highpass
            else:
                filter_types[i, j] = 3  # Bandpass
    
    # Plot filter type classification
    cmap = plt.cm.get_cmap('viridis', 4)
    sns.heatmap(filter_types, 
                xticklabels=[f"{t:.1f}" for t in tone_range],
                yticklabels=[f"{g:.1f}" for g in gain_range],
                cmap=cmap, annot=True, fmt="d")
    plt.xlabel("Tone Parameter")
    plt.ylabel("Gain Parameter")
    plt.title(f"Filter Type Classification for {PEDAL}")
    plt.gca().invert_yaxis()
    
    # Add colorbar labels
    colorbar = plt.gca().collections[0].colorbar
    colorbar.set_ticks([0.4, 1.2, 2.0, 2.8])
    colorbar.set_ticklabels(['Broadband', 'Lowpass', 'Highpass', 'Bandpass'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{PEDAL}_filter_type_classification.png"))
    plt.show()
    
    return {
        'centroid_shift': results_centroid,
        'low_cutoff': results_low_cutoff,
        'high_cutoff': results_high_cutoff,
        'q_factor': results_q_factor,
        'spectral_tilt': results_tilt,
        'filter_types': filter_types
    }


# Run the analysis
if __name__ == "__main__":
    results = evaluate_filter_grid()