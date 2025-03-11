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

def generate_test_signal(duration=1.0, sample_rate=SR):
    """Generate a clean sine wave test signal"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Create a 440 Hz sine wave (A4 note)
    return np.sin(2 * np.pi * 440 * t)

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

def calculate_thd(original, distorted):
    """
    Calculate Total Harmonic Distortion metric
    
    THD = sqrt(sum of power of harmonics / power of fundamental)
    """
    # Get frequency spectra
    orig_spectrum = np.abs(np.fft.rfft(original))
    dist_spectrum = np.abs(np.fft.rfft(distorted))
    
    # Find fundamental frequency (should be around 440 Hz bin)
    freq_bins = np.fft.rfftfreq(len(original), d=1/SR)
    fundamental_idx = np.argmax(orig_spectrum)
    fundamental_freq = freq_bins[fundamental_idx]
    
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

def evaluate_distortion_grid():
    """
    Evaluate distortion using a grid search over gain and tone parameters.
    Creates a heatmap of THD values.
    """
    # Generate clean test signal
    clean_signal = generate_test_signal()
    
    # Define parameter ranges
    gain_range = np.linspace(0, 5, 6)
    tone_range = np.linspace(0, 5, 6)
    
    # Initialize results matrix
    results = np.zeros((len(gain_range), len(tone_range)))
    
    # Perform grid search
    for i, gain in enumerate(gain_range):
        for j, tone in enumerate(tone_range):
            # Apply distortion
            distorted = guitar_overdrive(clean_signal, gain, tone)
            
            # Calculate THD
            thd = calculate_thd(clean_signal, distorted)
            
            # Store result
            results[i, j] = thd
    
    # Create heatmap
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(results, 
                   xticklabels=[f"{t:.1f}" for t in tone_range],
                   yticklabels=[f"{g:.1f}" for g in gain_range],
                   annot=True, fmt=".3f", cmap="viridis")
    
    ax.invert_yaxis()
    
    ax.set_xlabel("Tone Parameter")
    ax.set_ylabel("Gain Parameter")
    ax.set_title("Total Harmonic Distortion (THD) Across Parameter Space")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,"distortion_heatmap.png"))
    plt.show()
    
    return results

# Run the analysis
if __name__ == "__main__":
    results = evaluate_distortion_grid()

    
    # Additional visualization: Show waveforms for low, medium and high distortion
    clean = generate_test_signal(duration=0.01)  # Short sample for visualization
    
    fig, axes = plt.subplots(3, 1, figsize=(5, 5))
    
    # Low distortion
    low_dist = guitar_overdrive(clean, gain=0.5, tone=2.5)
    axes[0].plot(clean, label="Original")
    axes[0].plot(low_dist, label="Distorted", alpha=0.7)
    axes[0].set_title(f"Low Distortion (gain=0.5, tone=2.5, THD={calculate_thd(clean, low_dist):.3f})")
    axes[0].legend()
    
    # Medium distortion
    med_dist = guitar_overdrive(clean, gain=2.5, tone=2.5)
    axes[1].plot(clean, label="Original")
    axes[1].plot(med_dist, label="Distorted", alpha=0.7)
    axes[1].set_title(f"Medium Distortion (gain=2.5, tone=2.5, THD={calculate_thd(clean, med_dist):.3f})")
    axes[1].legend()
    
    # High distortion
    high_dist = guitar_overdrive(clean, gain=5.0, tone=2.5)
    axes[2].plot(clean, label="Original")
    axes[2].plot(high_dist, label="Distorted", alpha=0.7)
    axes[2].set_title(f"High Distortion (gain=5.0, tone=2.5, THD={calculate_thd(clean, high_dist):.3f})")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,"waveform_comparison.png"))
    plt.show()