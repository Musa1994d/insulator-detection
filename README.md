# insulator-detection
The goal of this project is to create an artificial neural network model using transfer learning to detect insulators in images. The yolov8n algorithm is used as a pre-trained model and is trained on a custom dataset including insulators.
The program is written in Google colab and you can reach it [here](https://github.com/Musa1994d/insulator-detection/blob/main/insulator_detect_yolo8n.ipynb).
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
First, import YOLO from ultralytics library:
```py
from ultralytics import YOLO
```
Next, load the pretrained model:
```py
model = YOLO('yolov8n.pt')  # load a pretrained model
```
Next, train the model on the dataset in 60 epochs:
```py
results = model.train(data='/content/insatance-segmentation-insulator-2/data.yaml', epochs=60, batch=8, imgsz=640)
```
Considering the number of images in our dataset is not large, we chose the batch value of 8.
Note that before running this training code cell, go to the "/content/insatance-segmentation-insulator-2/data.yaml" path in your google colab envirenment and open the 'data.yaml' file. Edit two pathes regarding 'train' and 'val'. Replace "/content/insatance-segmentation-insulator-2/train/images" with the path that already exists next to "train" and replace "/content/insatance-segmentation-insulator-2/valid/images" with the path that already exists next to "val". Then run the train code cell and wait for the training process to finish

Next, run this code to save the training results in your google colab envieronment to your google drive storage: 
```py
import shutil
shutil.copytree('/content/runs','/content/drive/MyDrive/"desired-path"',dirs_exist_ok=True)
```
Replace the desired path where you want to save the results to your drive with "desired-path"

## results
In [this folder](https://github.com/Musa1994d/insulator-detection/tree/main/results/weights) you can reach the trained weights. And [here](https://github.com/Musa1994d/insulator-detection/tree/main/results) you can get all the results. For example, some images predected in the validation step are given below:

![](https://github.com/Musa1994d/insulator-detection/blob/main/results/val_batch1_pred.jpg)


![](https://github.com/Musa1994d/insulator-detection/blob/main/results/val_batch0_pred.jpg)


And some of the training metrics are shown below:

![](https://github.com/Musa1994d/insulator-detection/blob/main/results/results.png)

As it can be seen, the amount of errors during training and validation is decreasing, and the values ​​of mAP50 and precision are increasing, which indicate the correctness of the training process.
