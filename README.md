# insulator-detection
The goal of this project is to create an artificial neural network model using transfer learning to detect insulators in images. The yolov8n algorithm is used as a pre-trained model and is trained on a custom dataset including insulators.
The program is written in Google colab and you can reach it [here](https://github.com/Musa1994d/insulator-detection/blob/main/insulator_detect_yolo8n.ipynb). The results of This project are in [this folder](https://github.com/Musa1994d/insulator-detection/tree/main/results)

## dataset
The dataset used in this project includes 917 images of insulators, 734 of which are considered as training data and 183 of which are considered as validation data. The size of the images is 640x640. this images are annotated in yolo format and are categorised in one class named 'insulator'.
You can see and download this dataset [here](https://drive.google.com/drive/folders/1ht-Rm8S9wrBUQepxTEXjVMHnGwsKJi_t?usp=sharing)

This dataset will be entered into the program environment from roboflow.com using the following code:
```py
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="eK5B9TqoU0cYD6XVQypg")
project = rf.workspace("pourya-shojaei").project("insatance-segmentation-insulator")
dataset = project.version(2).download("yolov8")
```
After running this snippet, the dataset will be downloaded and added to the content


## import and train yolov8n model
first, import YOLO from ultralytics library:
```py
from ultralytics import YOLO
```
next, load the pretrained model:
```py
model = YOLO('yolov8n.pt')  # load a pretrained model
```
next, train the model on the dataset in 60 epochs:
```py
results = model.train(data='/content/insatance-segmentation-insulator-2/data.yaml', epochs=60, batch=8, imgsz=640)
```
Considering the number of images in our database is not large, we chose the batch value of 8.
note that before running training code cell, go to the "" path in your google colab envirenment and open the 'data.yaml' file. edit 
