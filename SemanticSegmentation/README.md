[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6ndC2138)

# California Roll ~ Comparison of models for Semantic Segmentation of Satellite Images
This project aims to perform semantic segmentation of satellite images using various deep learning models. Our primary goal is to detect human settlements and electricity availability in regions based on satellite imagery data.
## Contributors

- [@HosunS](https://github.com/HosunS) - Hosun Song
- [@Revisha7](https://github.com/Revisha7) - Rebecca Park
- [@mlbusby](https://github.com/mlbusby) - Matthew Busby
- [@mikeyeunguci](https://github.com/mikeyeunguci) - Michael Yeung

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/PyTorchLightning-%23ffffff.svg?style=for-the-badge&logo=PyTorchLightning&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![wandb](https://img.shields.io/badge/wandb-%23ffffff.svg?style=for-the-badge&logo=wandb&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Dataset
The dataset used is the IEEE GRSS 2021 ESD dataset, which consists of 60 tiles of 800x800 pixels. Each tile includes 98 channels from different satellite images (Sentinel-1,Sentinel-2, Landsat 8, and VIIRS).
and groundtruth.tif for the groundtruth labels, as well as a groundtruth.png to help visualize.

### Satellite Data
We are provided with 4 different satellites and a multitude of bands to select.
* Sentinel-1: 2 channels (VV,VH)
* Sentinel-2: 12 channels
* Landsat 8: 11 channels
* VIIRS: 1 channel

### Semantic Labels
The dataset provides ground truth labels for four classes:
1. ![#ff0000](https://placehold.co/15x15/ff0000/ff0000.png) `Human settlements without electricity`
2. ![#0000ff](https://placehold.co/15x15/0000ff/0000ff.png) `No human settlements without electricity`
3. ![#ffff00](https://placehold.co/15x15/ffff00/ffff00.png) `Human settlements with electricity`
4. ![#b266ff](https://placehold.co/15x15/b266ff/b266ff.png) `No human settlements with electricity`


Through various testing, we decided to focus on using bands 2,3,4 from sentinel-2 and the VIIRS and MAXPROJECTION of VIIRS for our task of categorizing settlements. We chose the sentinel-2 bands 2,3,4 for the rgb composite which will help our model learn whether the location has settlments or not. We chose to use VIIRS and the MAXPROJECTION of VIIRS in order to help our model learn whether the locations have electricity or not by looking at the variations of light intensity through its sensors.

## Models
We used UNet, SegmentationCNN , and FCNResnetTransfer as our baseline models. We then went on to build on top of these baselines by including various techniques particularly dilated convolutions by implementing DilatedUNet and DeepLabV3.

### Dilated UNet
A variant of the original UNet architecture that incorporates dilated convolutions. Dilated convolutions expand the receptive field without losing resolution, allowing the model to capture multi-scale context information. This model helps the model better understand the spatial relationships in the image, and should help improve the segmentation performance, especially within the finer details in the images.

![image](https://github.com/cs175cv-s2024/final-project-california-roll/assets/117314672/dd97010d-d328-4375-aab2-bdad01b7edd3)



### DeepLabV3
DeepLabV3 is a deep learning architecture that is used for semantic segmentation. It also uses dilated convolutions to capture multi-scale context by adjusting the dilation rate in the convolutions, allowing it to control the resolution at the features are computed which is useful for segmenting objects at various scales. DeepLabV3 also includes an atrous spatial pyramid pooling module, which probes the incoming features with multiple atrous rates, which also is useful in capturing objects and image context at multiple scales.

![image](https://github.com/cs175cv-s2024/final-project-california-roll/assets/91280111/dfa2e1c0-6a6f-4c6f-b97e-bbd7e3ba04f3)


## Getting Started

### Virtual Environment & Installation

Ensure that you have the proper packages in your environment, create a virtual environment so that this project runs with the correct dependencies independently from other python projects. Follow these steps: 

Clone the repository and install the required packages provided in requirements.txt

```
git clone <repository_url>
cd <repository_directory>
python -m venv env
source env/bin/activate #for MacOS and Linux
.\env\Scripts\activate # on windows
pip install -r requirements.txt
```

### Dataset
Please download and unzip the `dfc2021_dse_train.zip` saving the `Train` directory into the `data/raw` directory. You do not need to worry about registering to get the data from the IEEE DataPort as we have already downloaded it for you.
The zip file is available at the following [url](https://drive.google.com/file/d/1mVDV9NkmyfZbkSiD5lkskv_MwOuYxiog/view?usp=drive_link).

### Weights and Biases
You can log all the neccessary information for model training and validation by creating an account with Weights and Biases. To create an account go to: [wandb.ai](http://wandb.ai).

## Usage

### Training the Models
Run the train.py script with the default ESDConfig values (utilities.py) or use the custom command line argument values to change the various parameters. The model results are printed in the console, but also charted and logged using weights and biases. There is also a sweep.yml file which will allow the hyperparamater sweeps of all or a single model, and run this with train_sweeps.py and argparse the location of the sweeps.yml file.

### Evaluating the Models
Like with training, evaluating is also a script and all you have to do is run the evaluate.py file with the path to the model checkpoint file. This script will train the specified model and run  a validation loop with that model. It will then obtain the validation satellite tiles and plot the raw RGB, restitched groundtruth, and the model's prediction to be able to visually see the model's accuracy at classifying the various regions in the original image.

## Project Components
- [Poster](https://docs.google.com/presentation/d/1HaFA0QZwawHGodLXnSeK04xb3zdg4L4i3USFN1d1z5U/edit?usp=sharing)
- [Slides](https://docs.google.com/presentation/d/1JHWEkZ9yWT2Po_yVCNRDVhaWBxeXA9iD1cC71E5KPG8/edit?usp=sharing)
- [Presenation](https://drive.google.com/file/d/1EhUe1Msjfo0iQ87RPGNfsGJY99xhmAfU/view?usp=sharing)
- [Techmemo](https://docs.google.com/document/d/1yDaUwrmZvIqkeDMgKA8F6f7W6lkIJUwsD0bXHgwIYxI/edit?usp=sharing)


## License
Distributed under the MIT License. See LICENSE for more information.


## Acknowledgments
The following are the papers and research we referred to when creating the new models:
- https://paperswithcode.com/method/deeplabv3
- https://paperswithcode.com/paper/dilated-unet-a-fast-and-accurate-medical
- https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
- https://github.com/tensorflow/models/blob/master/research/deeplab/README.md

