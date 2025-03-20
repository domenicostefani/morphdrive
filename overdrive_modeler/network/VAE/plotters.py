import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from utils import DBmetadataRenamer

def extract_spectrogram(audio, sr=32000):
    extracted_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, fmax=sr//2)
    log_spectrogram = librosa.power_to_db(extracted_spectrogram, ref=np.max)
    return log_spectrogram

def plot_spectrograms_to_wandb(model_output, original_audio, sr):
    import wandb

    predicted = model_output.squeeze().cpu().detach().numpy()
    original = original_audio.squeeze().cpu().detach().numpy()

    predicted_spectrogram = extract_spectrogram(predicted, sr)
    original_spectrogram = extract_spectrogram(original, sr)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(predicted_spectrogram, aspect='auto', origin='lower', cmap='magma')
    plt.title('Predicted Spectrogram')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(original_spectrogram, aspect='auto', origin='lower', cmap='magma')
    plt.title('Original Spectrogram')
    plt.colorbar()

    wandb.log({"Spectrograms": wandb.Image(plt)})
    plt.close()


def load_audio_to_wandb(original, predicted, sr=32000):
    original = original.squeeze().cpu().detach().numpy()
    predicted = predicted.squeeze().cpu().detach().numpy()
    wandb.log({
        "original_audio": wandb.Audio(original, caption="Original Audio", sample_rate=sr),
        "predicted_audio": wandb.Audio(predicted, caption="Predicted Audio", sample_rate=sr)
    })


def normalize_coordinates(coords):
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min)
    coords[:, 1] = (coords[:, 1] - y_min) / (y_max - y_min)
    return coords


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def normalize_coordinates(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def visualize_latents(reduction, X_transformed, y, image_path, label_type="pedal", mode="2D",figsize=(10, 10), show_xy_titles=False, show_title=True, show_ticks=True, show_legend=True):
    fig = plt.figure(figsize=figsize)
    if mode == "3D":
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    if label_type == "pedal":
        labels = y["label"]
    elif label_type == "gain":
        labels = y["gain"]
    elif label_type == "tone":
        labels = y["tone"]

    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)

    if num_labels < 10:
        cmap = sns.color_palette()
        # Skip green
        rng = [i+1 if i >= 2 else i for i in range(num_labels)]
        rng = [2 if i == 10 else i for i in rng] # Place green at the end
        colors = [cmap[i] for i in rng]
    else:
        colors = [
            "#FF5733",  # Deep Orange-Red
            "#FFA500",  # Bright Orange
            "#FFC300",  # Warm Yellow-Orange
            "#B8860B",  # Dark Goldenrod
            "#4682B4",  # Steel Blue
            "#87CEFA",  # Light Sky Blue
            "#1E3A8A",  # Dark Navy Blue
            "#9370DB",  # Medium Purple
            "#DDA0DD",  # Light Purple (Plum)
            "#FF69B4",  # Hot Pink
            "#DB7093",  # Muted Pink (Rosy Brown)
            "#BA55D3",  # Medium Orchid
            "#D8BFD8",  # Soft Lavender
            "#8B4513",  # Saddle Brown
            "#808080",  # Neutral Gray
            "#B0C4DE",  # Light Steel Blue (Grayish Blue)
            "#6A5ACD",  # Slate Blue
            "#4B0082",  # Indigo
            "#8B0000",  # Dark Red
            "#DC143C",  # Crimson
            "#CD5C5C",  # Indian Red
            "#FF6347",  # Tomato Red
            "#E9967A",  # Light Coral
            "#FFDAB9",  # Peach Puff (Soft Peach)
            "#F4A460",  # Sandy Brown
            "#D2691E",  # Chocolate
            "#5F4B32"   # Deep Brown
        ]

    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    datasetMetadataRenamer = DBmetadataRenamer()

    label_names = {label: datasetMetadataRenamer.datasetname2shortname(label) if label_type == "pedal" else str(label) for label in unique_labels}

    for label in unique_labels:
        indices = (labels == label)
        if mode == "3D":
            ax.scatter(X_transformed[indices, 0], X_transformed[indices, 1], X_transformed[indices, 2],
                        color=[label_to_color[label]], label=label_names[label], s=125, edgecolors='w')
        else:
            ax.scatter(X_transformed[indices, 0], X_transformed[indices, 1],
                        color=[label_to_color[label]], label=label_names[label], s=125, edgecolors='w')

    if show_title:
        ax.set_title(f"{reduction} on Latents ({label_type.capitalize()})")
    if show_xy_titles:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        if mode == "3D":
            ax.set_zlabel("Component 3")

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        if mode == "3D":
            ax.set_zticks([])
    
    if show_legend:
        # The legend should be on the right side of the plot, outside the plot area
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=label_type.capitalize())

    plt.tight_layout()
    plt.savefig(image_path, bbox_inches='tight',pad_inches = 0)
    print(f"Saved plot to {image_path}")
    # Png
    image_path = image_path.replace(".pdf", ".png")
    plt.savefig(image_path, bbox_inches='tight',pad_inches = 0)
    print(f"Saved plot to {image_path}")
    plt.show()

def pca_on_latents(latents_csv_path, csv_path, image_path, label_type="pedal", mode="2D",figsize=(10, 10), show_xy_titles=False, show_title=True, show_ticks=True, show_legend=True):
    df = pd.read_csv(latents_csv_path)
    df["latents"] = df["latents"].apply(eval)  
    X = np.array(df["latents"].to_list())
    y = df[["label", "gain", "tone"]]

    n_components = 3 if mode == "3D" else 2
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_pca = normalize_coordinates(X_pca)
    df["coords"] = X_pca.tolist()  

    df["label_name"] = df["label"]
    df = df[["label_name", "gain", "tone", "latents", "coords"]]
    df.to_csv(csv_path, index=False)
    
    visualize_latents("PCA", X_pca, y, image_path, label_type, mode, figsize, show_xy_titles, show_title, show_ticks, show_legend)

def tsne_on_latents(latents_csv_path, csv_path, image_path, label_type="pedal", mode="2D", figsize=(10, 10), show_xy_titles=False, show_title=True, show_ticks=True, show_legend=True):
    df = pd.read_csv(latents_csv_path)
    df["latents"] = df["latents"].apply(eval) 
    X = np.array(df["latents"].to_list())
    y = df[["label", "gain", "tone"]]

    n_components = 3 if mode == "3D" else 2
    tsne = TSNE(n_components=n_components, perplexity=min(80, len(y)-1), n_iter=2000, random_state=42)
    X_tsne = tsne.fit_transform(X)
    X_tsne = normalize_coordinates(X_tsne)
    df["coords"] = X_tsne.tolist()  

    df["label_name"] = df["label"]
    df = df[["label_name", "gain", "tone", "latents", "coords"]]
    df.to_csv(csv_path, index=False)

    visualize_latents("TSNE", X_tsne, y, image_path, label_type, mode, figsize, show_xy_titles, show_title, show_ticks, show_legend)

if __name__ == "__main__":
    print("Plotters-Script")
    import argparse, os
    parser = argparse.ArgumentParser()
    # One argument, mandatory: path to the latents csv file
    parser.add_argument("latents_csv_path", type=str, help="Path to the latents csv file")
    # One argument "type" that defaults to tsne
    parser.add_argument("--compute", type=str, default=None, help="Type of reduction (pca or tsne)")
    parser.add_argument("--figsize", '-f', type=float, nargs=2, default=(5, 5), help="Figure size")
    parser.add_argument('--show-legend','-l', action='store_true', help='Show legend on the plot')

    args = parser.parse_args()
    figsize = tuple(args.figsize)

    show_legend = args.show_legend

    if args.compute == "pca":
        pca_on_latents(args.latents_csv_path, csv_path="pca_latents.csv", image_path="pca_latents.pdf", figsize=figsize, show_title=False, show_xy_titles=False, show_ticks=False, show_legend=False)
        print(f'PCA on latents completed. Results saved to ./pca_latents.csv and ./pca_latents.pdf')
    elif args.compute == "tsne":
        tsne_on_latents(args.latents_csv_path, csv_path="tsne_latents.csv", image_path="tsne_latents.pdf", figsize=figsize, show_title=False, show_xy_titles=False, show_ticks=False, show_legend=False)
        print(f'TSNE on latents completed. Results saved to ./tsne_latents.csv and ./tsne_latents.pdf')
    elif args.compute is None:
        print("WARNING: NOT computing any reduction, but only visualizing precomputed reduction")
        df = pd.read_csv(args.latents_csv_path)
        if "label_name" not in df.columns:
            raise ValueError("You may be trying to visualize latents from a temporary csv with no precomputed reduction. Please run the script with --compute pca or --compute tsne")

        X_tsne = np.array(df["coords"].str.strip('[]').str.split(',').apply(lambda x: [float(x[0]), float(x[1]) if len(x) > 1 else 0]).to_list())

        df.rename(columns={"label_name": "label"}, inplace=True)
        y = df[["label", "gain", "tone"]]
        image_path = os.path.splitext(args.latents_csv_path)[0] + "_latents.pdf"
        label_type = "pedal"
        mode = "2D"
        show_xy_titles = False
        show_title = False
        show_ticks = False
   
        visualize_latents("TSNE", X_tsne, y, image_path, label_type, mode, figsize, show_xy_titles, show_title, show_ticks, show_legend)
    else:
        print("Invalid type. Please choose either 'pca' or 'tsne'.")
        
    pass