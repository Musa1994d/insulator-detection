# insulator-detection
The goal of this project is to create an artificial neural network model using transfer learning to detect insulators in images. The yolo8n algorithm is used as a pre-trained model and is trained on a custom dataset including insulators.
The program is written in Google colab and you can reach it [here](https://github.com/Musa1994d/insulator-detection/blob/main/insulator_detect_yolo8n.ipynb). The results of This project are in [this folder](https://github.com/Musa1994d/insulator-detection/tree/main/results)

## dataset
The dataset used in this project includes 917 images of insulators, 734 of which are considered as training data and 183 of which are considered as validation data. You can see and download this dataset [here](https://drive.google.com/drive/folders/1ht-Rm8S9wrBUQepxTEXjVMHnGwsKJi_t?usp=sharing)
This dataset will be entered into the program environment from roboflow.com using the following code:
```py
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="eK5B9TqoU0cYD6XVQypg")
project = rf.workspace("pourya-shojaei").project("insatance-segmentation-insulator")
dataset = project.version(2).download("yolov8")
```
