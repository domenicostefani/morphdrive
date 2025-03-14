import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import librosa
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory

SR=48000

OUTDIR = 'test_plots'
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

def generate_sine_sweep(duration=1.0, sample_rate=SR, f_start=15, f_end=24000):
    """Generate a logarithmic sine sweep from f_start to f_end Hz"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Logarithmic sweep formula
    k = np.exp(np.log(f_end/f_start) / duration)
    phase = 2 * np.pi * f_start * ((k**t - 1) / np.log(k))
    sweep = np.sin(phase)
    
    return sweep, t


def guitar_overdrive(audio, gain=1.0, tone=3.0):
    """
    Apply overdrive distortion effect to audio signal
    
    Parameters:
    - audio: Input audio signal
    - gain: Distortion amount (0.0 to 5.0)
    - tone: Tone control (0.0 to 5.0, higher values = more treble)
    
    Returns:
    - Processed audio signal
    """
    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Pre-gain
    audio = audio * (1 + gain * 3)
    
    # Soft clipping function (tanh provides a nice tube-like overdrive)
    audio = np.tanh(audio)
    
    # Apply tone control (simple low-pass filter)
    # Lower tone values = more bass (lower cutoff)
    cutoff = 1000 + tone * 1000  # 1kHz to 6kHz
    b, a = signal.butter(2, cutoff / 22050, btype='low')
    audio = signal.filtfilt(b, a, audio)
    
    # Output level compensation
    audio = audio * 0.7
    
    return audio

def calculate_spectral_difference(original, distorted):
    """
    Calculate a measure of distortion based on spectral difference
    across the entire frequency range of the sweep
    """
    # Get frequency spectra
    orig_spectrum = np.abs(np.fft.rfft(original))
    dist_spectrum = np.abs(np.fft.rfft(distorted))
    
    # Normalize spectra
    orig_spectrum = orig_spectrum / np.max(orig_spectrum)
    dist_spectrum = dist_spectrum / np.max(dist_spectrum)
    
    # Calculate difference between spectra
    # We can use various metrics here:
    
    # 1. Mean squared error between spectra
    mse = np.mean((dist_spectrum - orig_spectrum)**2)
    
    # 2. Energy ratio of added harmonics
    energy_ratio = np.sum(dist_spectrum**2) / np.sum(orig_spectrum**2)
    
    # 3. Spectral centroid shift (change in brightness)
    freq_bins = np.fft.rfftfreq(len(original), d=1/SR)
    orig_centroid = np.sum(freq_bins * orig_spectrum) / np.sum(orig_spectrum)
    dist_centroid = np.sum(freq_bins * dist_spectrum) / np.sum(dist_spectrum)
    centroid_shift = dist_centroid / orig_centroid
    
    return {
        'mse': mse,
        'energy_ratio': energy_ratio,
        'centroid_shift': centroid_shift
    }

def evaluate_distortion_grid():
    """
    Evaluate distortion using a grid search over gain and tone parameters.
    Creates heatmaps of different distortion metrics using a sine sweep.
    """
    # Generate sine sweep test signal
    clean_signal, time_array = generate_sine_sweep(duration=2.0, f_start=80, f_end=8000)
    
    # Define parameter ranges
    gain_range = np.linspace(0, 5, 6)
    tone_range = np.linspace(0, 5, 6)
    
    # Initialize results matrices
    results_mse = np.zeros((len(gain_range), len(tone_range)))
    results_energy = np.zeros((len(gain_range), len(tone_range)))
    results_centroid = np.zeros((len(gain_range), len(tone_range)))
    
    # Perform grid search
    for i, gain in enumerate(gain_range):
        for j, tone in enumerate(tone_range):
            # Apply distortion
            distorted = guitar_overdrive(clean_signal, gain, tone)
            
            # Calculate metrics
            metrics = calculate_spectral_difference(clean_signal, distorted)
            
            # Store results
            results_mse[i, j] = metrics['mse']
            results_energy[i, j] = metrics['energy_ratio']
            results_centroid[i, j] = metrics['centroid_shift']
    
    # Create heatmaps for each metric
    plt.figure(figsize=(6, 5))
    
    # MSE Heatmap
    # plt.subplot(1, 3, 1)
    # sns.heatmap(results_mse, 
    #             xticklabels=[f"{t:.1f}" for t in tone_range],
    #             yticklabels=[f"{g:.1f}" for g in gain_range],
    #             annot=True, fmt=".3f", cmap="viridis")
    # plt.xlabel("Tone Parameter")
    # plt.ylabel("Gain Parameter")
    # plt.title("Spectral Difference (MSE)")
    # plt.gca().invert_yaxis()
    
    # Energy Ratio Heatmap
    # plt.subplot(1, 3, 2)
    sns.heatmap(results_energy, 
                xticklabels=[f"{t:.1f}" for t in tone_range],
                yticklabels=[f"{g:.1f}" for g in gain_range],
                annot=True, fmt=".3f", cmap="viridis")
    plt.xlabel("Tone Parameter")
    plt.ylabel("Gain Parameter")
    plt.title("Energy Ratio (Higher = More Distortion)")
    plt.gca().invert_yaxis()
    
    # Centroid Shift Heatmap
    # plt.subplot(1, 3, 3)
    # sns.heatmap(results_centroid, 
    #             xticklabels=[f"{t:.1f}" for t in tone_range],
    #             yticklabels=[f"{g:.1f}" for g in gain_range],
    #             annot=True, fmt=".3f", cmap="viridis")
    # plt.xlabel("Tone Parameter")
    # plt.ylabel("Gain Parameter")
    # plt.title("Spectral Centroid Shift (Brightness Change)")
    # plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,"distortion_sweep_heatmap.png"))
    plt.show()
    
    # # Visualize spectrograms for low, medium, and high distortion
    # plt.figure(figsize=(18, 12))
    
    # # Original sweep spectrogram
    # plt.subplot(4, 1, 1)
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(clean_signal)), ref=np.max)
    # librosa.display.specshow(D, y_axis='log', x_axis='time', sr=SR)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Original Sine Sweep Spectrogram')
    
    # # Low distortion
    # plt.subplot(4, 1, 2)
    # low_dist = guitar_overdrive(clean_signal, gain=0.5, tone=2.5)
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(low_dist)), ref=np.max)
    # librosa.display.specshow(D, y_axis='log', x_axis='time', sr=SR)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Low Distortion Spectrogram (gain=0.5, tone=2.5)')
    
    # # Medium distortion
    # plt.subplot(4, 1, 3)
    # med_dist = guitar_overdrive(clean_signal, gain=2.5, tone=2.5)
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(med_dist)), ref=np.max)
    # librosa.display.specshow(D, y_axis='log', x_axis='time', sr=SR)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Medium Distortion Spectrogram (gain=2.5, tone=2.5)')
    
    # # High distortion
    # plt.subplot(4, 1, 4)
    # high_dist = guitar_overdrive(clean_signal, gain=5.0, tone=2.5)
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(high_dist)), ref=np.max)
    # librosa.display.specshow(D, y_axis='log', x_axis='time', sr=SR)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('High Distortion Spectrogram (gain=5.0, tone=2.5)')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTDIR,"distortion_spectrograms.png"))
    # plt.show()
    
    return results_mse, results_energy, results_centroid

# Run the analysis
if __name__ == "__main__":
    results = evaluate_distortion_grid()