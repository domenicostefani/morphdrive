import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
import librosa
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def extract_spectrogram(audio, sr=32000):
    extracted_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, fmax=sr//2)
    log_spectrogram = librosa.power_to_db(extracted_spectrogram, ref=np.max)
    return log_spectrogram

def plot_spectrograms_to_wandb(model_output, original_audio, sr):

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

def visualize_latents(reduction, X_transformed, y, index_to_label, image_path, label_type="pedal", mode="2D"):
    fig = plt.figure(figsize=(10, 10))
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

    cmap = plt.get_cmap("tab10" if num_labels <= 10 else "hsv")
    colors = [cmap(i / num_labels) for i in range(num_labels)]
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    label_names = {label: index_to_label[label] if label_type == "pedal" else str(label) for label in unique_labels}

    for label in unique_labels:
        indices = (labels == label)
        if mode == "3D":
            ax.scatter(X_transformed[indices, 0], X_transformed[indices, 1], X_transformed[indices, 2],
                        color=[label_to_color[label]], label=label_names[label], s=100)
        else:
            ax.scatter(X_transformed[indices, 0], X_transformed[indices, 1],
                        color=[label_to_color[label]], label=label_names[label], s=100)

    ax.set_title(f"{reduction} on Latents ({label_type.capitalize()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if mode == "3D":
        ax.set_zlabel("Component 3")
    
    plt.legend()
    plt.savefig(image_path)
    plt.show()

def pca_on_latents(index_to_label, latents_csv_path, csv_path, image_path, label_type="pedal", mode="2D"):
    df = pd.read_csv(latents_csv_path)
    df["latents"] = df["latents"].apply(eval)  
    X = np.array(df["latents"].to_list())
    y = df[["label", "gain", "tone"]]

    n_components = 3 if mode == "3D" else 2
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_pca = normalize_coordinates(X_pca)
    df["coords"] = X_pca.tolist()  

    df["label_name"] = df["label"].map(index_to_label)
    df = df[["label_name", "gain", "tone", "latents", "coords"]]
    df.to_csv(csv_path, index=False)
    
    visualize_latents("PCA", X_pca, y, index_to_label, image_path, label_type, mode)

def tsne_on_latents(index_to_label, latents_csv_path, csv_path, image_path, label_type="pedal", mode="2D"):
    df = pd.read_csv(latents_csv_path)
    df["latents"] = df["latents"].apply(eval) 
    X = np.array(df["latents"].to_list())
    y = df[["label", "gain", "tone"]]

    n_components = 3 if mode == "3D" else 2
    tsne = TSNE(n_components=n_components, perplexity=min(80, len(y)-1), n_iter=2000, random_state=42)
    X_tsne = tsne.fit_transform(X)
    X_tsne = normalize_coordinates(X_tsne)
    df["coords"] = X_tsne.tolist()  

    df["label_name"] = df["label"].map(index_to_label)
    df = df[["label_name", "gain", "tone", "latents", "coords"]]
    df.to_csv(csv_path, index=False)
    
    visualize_latents("TSNE", X_tsne, y, index_to_label, image_path, label_type, mode)
