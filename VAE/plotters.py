import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
import librosa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def extract_spectrogram(audio, sr=32000):
    extracted_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, fmax=sr//2)
    log_spectrogram = librosa.power_to_db(extracted_spectrogram, ref=np.max)
    return log_spectrogram

def plot_spectrograms_to_wandb(model_output, original_audio):

    predicted = model_output.squeeze().cpu().detach().numpy()
    original = original_audio.squeeze().cpu().detach().numpy()

    predicted_spectrogram = extract_spectrogram(predicted)
    original_spectrogram = extract_spectrogram(original)

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



def pca_on_latents(index_to_pedal_label, latents_csv_path, folder_path):
    df = pd.read_csv(latents_csv_path)
    
    X = df.drop(columns=["label", "gain", "tone"])

    for type in ['pedal', 'gain', 'tone']:
        if type == "pedal":
            y = df["label"]
        elif type == "gain":
            y = df["gain"]
        elif type == "tone":
            y = df["tone"]
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        unique_labels = np.unique(y)
        num_labels = len(unique_labels)
        
        if type == "pedal":
            cmap = plt.get_cmap("tab10" if num_labels <= 10 else "hsv")  
            colors = [cmap(i / num_labels) for i in range(num_labels)]
        else:
            cmap = plt.get_cmap("coolwarm")  
            colors = [cmap(i / (num_labels - 1)) for i in range(num_labels)]
        
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        if type == "pedal":
            label_names = {label: index_to_pedal_label[label] for label in unique_labels}  
        else:
            label_names = {label: str(label) for label in unique_labels} 
        
        plt.figure(figsize=(10, 10))
        
        for label in unique_labels:
            indices = (y == label)
            plt.scatter(X_pca[indices, 0], X_pca[indices, 1], color=[label_to_color[label]], label=label_names[label])
        
        plt.legend()
        plt.title(f"PCA on Latents ({type.capitalize()})")
        plt.savefig(f'{folder_path}/PCA_latents_VAE_{type}.png')



def tsne_on_latents(index_to_pedal_label, latents_csv_path, folder_path):
    df = pd.read_csv(latents_csv_path)

    X = df.drop(columns=["label", "gain", "tone"])
    tsne = TSNE(n_components=2, perplexity=80, n_iter=2000)
    X_tsne = tsne.fit_transform(X)

    for type in ['pedal', 'gain', 'tone']:
        if type == "pedal":
            y = df["label"]
        elif type == "gain":
            y = df["gain"]
        elif type == "tone":
            y = df["tone"]
        
        unique_labels = np.unique(y)
        num_labels = len(unique_labels)
        
        if type == "pedal":
            cmap = plt.get_cmap("tab10" if num_labels <= 10 else "hsv")
            colors = [cmap(i / num_labels) for i in range(num_labels)]
        else:
            cmap = plt.get_cmap("coolwarm")
            colors = [cmap(i / (num_labels - 1)) for i in range(num_labels)]
        
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        if type == "pedal":
            label_names = {label: index_to_pedal_label[label] for label in unique_labels}
        else:
            label_names = {label: str(label) for label in unique_labels}
        
        plt.figure(figsize=(10, 10))
        
        for label in unique_labels:
            indices = (y == label)
            plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], color=[label_to_color[label]], label=label_names[label])
        
        plt.legend()
        plt.title(f"t-SNE on Latents ({type.capitalize()})")
        plt.savefig(f'{folder_path}/TSNE_latents_VAE_{type}.png')