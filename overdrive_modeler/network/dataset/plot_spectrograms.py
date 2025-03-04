# import librosa
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from glob import glob
# os.chdir(os.path.dirname(__file__))

# def plot_spectrograms(wav_file, ax):
#     y, sr = librosa.load(wav_file, sr=48000)
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)

#     librosa.display.specshow(D, sr=sr,
#                              x_axis='time',
#                              y_axis='linear',
#                              hop_length=512)
#     # plt.colorbar(format='%+2.0f dB')
#     # Remove colorbar
#     # Change y-axis from Hz to kHz
#     ax.set_yticks(np.arange(0, ax.get_ylim()[1], 5000), np.arange(0, ax.get_ylim()[1]/1000, 5))
#     ax.set_yticklabels(np.arange(0, ax.get_ylim()[1]/1000, 5).astype(int))
#     ax.set_ylabel('Frequency (kHz)')

#     ax.set_xlabel('Time (s)')
#     ax.set_xticks(np.arange(0, ax.get_xlim()[1], 1))
#     ax.set_xticklabels(np.arange(0, ax.get_xlim()[1], 1).astype(int))

#     ax.set_title(os.path.basename(wav_file))

#     # fig.title('Spectrogram')
#     # fig.tight_layout()
#     # fig.savefig(output_file)


# if __name__ == "__main__":

#     N_TOPLOT = 2

#     interesting_files = glob("*/s*_*_g5_t5*.wav")
#     interesting_files = interesting_files[:N_TOPLOT] # Limit to N_TOPLOT files

#     base_figsize = (4, 3.5)
#     fig, axs = plt.subplots(1, N_TOPLOT, figsize=(base_figsize[0]*N_TOPLOT, base_figsize[1]))

#     for fpath,ax in zip(interesting_files,axs):
#         assert os.path.exists(fpath), "File not found"
#         # imgname = os.path.splitext(os.path.basename(fpath))[0]+'.png'
#         plot_spectrograms(fpath, ax)

#         # Create random barpplot to test


#     plt.tight_layout()
#     plt.show()


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
os.chdir(os.path.dirname(__file__))


folder_brand_model_map = {
    "kingoftone": ("AnalogMan","King of Tone"),
    "honeybee": ("Bearfoot","Honey Bee"),
    # "?": ("Beetronix","Beehive"),
    # "?": ("Bogner","Wessex"),
    "bluesdriver": ("Boss","Blues Driver"),
    "gladio": ("Cornerstone","Gladio"),
    "ss2": ("Cornish","SS2"),
    "theelements": ("Dr Scientist","The Elements"),
    "ocd": ("Fulltone","OCD"),
    "zendrive": ("Hermida Audio","Zendrive"),
    "precisiondrive": ("Horizon Devices","Precision Drive"),
    "tubedreamer": ("Jam Pedals","Tube Dreamer"),
    "ktr": ("Klon","KTR"),
    # "?": ("Lichtlaerm Audio","Acquaria"),
    "bigfella": ("Lunastone","Big Fella"),
    "chime": ("Pettyjohn","Chime"),
    # "?": ("Studio Daydream","Fetbox"),
    # "?": ("SviSound","Overzoid"),
    "dumkudo": ("Tanabe","Dumkudo"),
    "janray": ("Vemuram","Jan Ray"),
    "silkdrive": ("Vox","Silk Drive"),
    # "?": ("Way Huge","Red Llama"),
    # "?": ("Wampler","Mofetta"),
    "souldriven": ("Xotic","Soul Driven"),
}


def plot_spectrograms(wav_file, ax, is_leftmost=False, title="auto"):
    y, sr = librosa.load(wav_file, sr=48000)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)

    img = librosa.display.specshow(D, sr=sr,
                             x_axis='time',
                             y_axis='linear',
                             hop_length=512,
                             ax=ax)  # Make sure to pass the ax parameter here
    
    # Get the current y-axis limit to set the ticks correctly
    y_max = ax.get_ylim()[1]
    tick_interval = 5000
    num_ticks = int(y_max / tick_interval) + 1
    
    ax.set_yticks([i * tick_interval for i in range(num_ticks)])
     # Only add y-axis labels and title for the leftmost subplot
    if is_leftmost:
        ax.set_yticklabels([str(i * 5) for i in range(num_ticks)])
        ax.set_ylabel('Frequency (kHz)')
    else:
        ax.set_yticklabels([])  # Hide y-tick labels for non-leftmost subplots
        ax.set_ylabel('')


    # Get the current x-axis limit to set the ticks correctly
    x_max = ax.get_xlim()[1]
    tick_interval = 1.0
    num_ticks = int(x_max / tick_interval) + 1
    
    ax.set_xticks([i * tick_interval for i in range(num_ticks)])
    ax.set_xticklabels([str(i) for i in range(num_ticks)])
    ax.set_xlabel('Time (s)')
    
    if title == None:
        pass
    elif title == "auto":
        ax.set_title(os.path.basename(wav_file))
    else:
        ax.set_title(title)
    
    
    return img  # Return the image to prevent garbage collection


if __name__ == "__main__":
    N_TOPLOT = 4

    interesting_files = sorted(glob("*/s*_*_g5_t5*.wav"))
    interesting_files.sort(key=lambda x: os.path.basename(x))  # Sort by filename
    print("Interesting files:", "\n".join(interesting_files))
    interesting_files = interesting_files[:N_TOPLOT]  # Limit to N_TOPLOT files

    base_figsize = (3.5, 3.5)
    
    # Create figure and axes array before the loop
    fig, axs = plt.subplots(1, N_TOPLOT, figsize=(base_figsize[0]*N_TOPLOT, base_figsize[1]))
    
    axs = [axs] if N_TOPLOT == 1 else axs # Make sure axs is always iterable even with one subplot
    
    images = []  # Store references to the images
    
    for i, (fpath, ax) in enumerate(zip(interesting_files, axs)):
        assert os.path.exists(fpath), f"File not found: {fpath}"

        mapentry = folder_brand_model_map[os.path.basename(os.path.dirname(fpath))]
        curgain = os.path.basename(fpath).split("_")[2].lstrip("g")
        curtone = os.path.basename(fpath).split("_")[3].lstrip("t").rstrip(".wav")
        curtitle = f"{mapentry[0]} {mapentry[1]}, Gain {curgain}, Tone {curtone}"
        img = plot_spectrograms(fpath, ax, is_leftmost=(i == 0), title = curtitle)
        images.append(img)  # Keep a reference to prevent garbage collection
    
    # Add a colorbar to the figure
    # fig.colorbar(images[0], ax=axs, format='%+2.0f dB', shrink=0.8, label='Amplitude (dB)')
    
    plt.tight_layout()
    plt.savefig('sweep_spectrograms.png', dpi=200)
    plt.savefig('sweep_spectrograms.jpg', dpi=300)
    plt.savefig('sweep_spectrograms.pdf')
    # plt.show()