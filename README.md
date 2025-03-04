DAFX2025 Morph Overdrive Pedals Project
---

### Dataset

Examples of output signals for different pedals when the input signal is sine sweep.

`overdrive_modeler/network/dataset/sweep_spectrograms.jpg`  
![overdrive_modeler/network/dataset/sweep_spectrograms.jpg](overdrive_modeler/network/dataset/sweep_spectrograms.jpg)

### Recording Robot:

Pictures of the robot created and used to record the dataset.  
Two stepper motors are used to control the position of the gain and tone knobs on the pedal, which is fixed to the base with velcro straps. The motors are controlled by a Pure Data patch via an Arduino board and serial communication. Audio I/O and recording are handled in the same patch.


<div style="width: 70%; overflow: hidden;">
    <img src="docs/images/IMG_1128.jpg" 
    style="width: 100%;   margin: -50px 0 -120px 0">
</div>
  

<div style="width: 70%; overflow: hidden;">
    <img src="docs/images/IMG_1123.jpg" 
    style="width: 100%;   margin: -50px 0 -120px 0">
</div>

![Puredata patch](docs/images/puredata.png)


<!-- ### Neural Modeler -->

### GUI
![docs/images/gui_two_pedals.png](docs/images/gui_two_pedals.png)



<!-- <div style="width: 70%; overflow: hidden;">
    <img src="docs/images/IMG_1122.jpg" 
    style="width: 100%;   margin: -150px 0 -20px 0">
</div> -->


<!-- ![Robot Prototype](docs/images/IMG_1122.jpg)  -->


## Folders
<!-- - `robotic_database_recorder` - contains the code for the robotic database recorder -->
`docs` - documentation
`overdrive_modeler`  - contains the code for the neural network modeler and GUI
`robotic_database_recorder` -  contains the code for the robotic database recorder






Francesco Ardan Dal Rì  
Domenico Stefani