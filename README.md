<div align="center">

# Morphdrive: Latent Conditioning for Cross-Circuit Effect Modeling

<div align="center" style="margin-bottom:.5rem">
  <!-- <p style="font-size: 1.3em; margin-bottom: 0;"> -->
    <a href="https://github.com/return-nihil"> Ardan Dal Rì</a>,
    <a href="http://www.domenicostefani.com"> Domenico Stefani</a>, <br>
    Luca Turchet,
    Nicola Conci
      <!-- </p> -->
</div>
</div>

<div align="center" style="margin-bottom:2rem">
<i>University of Trento, Italy</i>
</div>

<center>
<div style="width: 100%">
<img src="docs/images/morphdrive-arch.svg" style="width: 38%;">
<img src="docs/images/morphdrive_gui.png"  style="width: 38%;">
</div>

*Accompanying material for "MorphDrive: Latent Conditioning for Cross-Circuit Effect Modeling and a Parametric Audio Dataset of Analog Overdrive Pedals"*
</center>

<div align="center">

### 🎵🎧 **[>>Website<<](https://www.domenicostefani.com/morphdrive/)** 🎧🎵  
<i>(with audio samples and more)</i>
</div>

<!-- ## Website with Examples

<a href="https://domenicostefani.com/morphdrive" style="margin-bottom:4 rem">www.domenicostefani.com/morphdrive</a> -->

## Abstract

<div align="justified">
We present an approach to the neural modeling of overdrive guitar pedals with conditioning from a cross-circuit and cross-setting latent space. The resulting networks model the behavior of multiple overdrive pedals across different settings, offering continuous morphing between real configurations and hybrid behaviors. Compact conditioning spaces are obtained through unsupervised training of a variational autoencoder with adversarial training, resulting in accurate reconstruction performance across different sets of pedals. We then compare three Hyper-Recurrent architectures for processing, including dynamic and static HyperRNNs, and a smaller model for real-time processing. Additionally, we present a new open dataset including recordings of 27 analog overdrive pedals, each with 36 gain and tone parameter combinations totaling over 97 hours of recordings. Precise parameter setting was achieved through a custom recording robot.</div>




## Folders
<!-- - `robotic_database_recorder` - contains the code for the robotic database recorder -->
`docs` - documentation  
`overdrive_modeler`  - contains the code for the neural network modeler, GUI, and evaluation  
`robotic_database_recorder` -  contains the code for the robotic database recorder  

<center>
<img src="docs/images/robot-gif.gif"  style="width: 78%;">
</center>