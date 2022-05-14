# DAT550: Deepfake detection challenge

### Group members
- Henrik Skulevold, University of Stavanger, Norway
- Vegard Matre, University of Stavanger, Norway
- Ådne Øvrebø, University of Stavanger, Norway

### Description
In this information age, with the raise of computer knowledge and
internet widely available, fake news and videos have become a big-
ger concern the later years. Fake news and videos could be used by
people to spread false information to influence people a certain way.
With AI and better tools such fake news and videos are becoming
harder and harder to expose. Hence, in this project we will try to
create an model that can detect fake videos. For training and testing
this model we will be using a dataset from DeepFake Detection
challenge at Kaggle. This is a large dataset containing both fake and
real videos along with metadata. This is implemented by creating a model that will try to classify the
video to either FAKE or REAL.

## Sample data
This repository comes with a sample dataset. The dataset is
available at https://www.kaggle.com/c/deepfake-detection-challenge/data. We have also provided a dataset containing 125 gigabytes of preceessed data dat on onedrive: https://1drv.ms/u/s!AsubJuTcblBEj8VRxK0X5v4FcZ6uMA?e=kXGuiC. The dataset contains both real and fake videos, splitted into train, test and validation.

## Project structure
The project is divided into three main parts:
- Data preprocessing
    - [preprocessing.ipynb](preprocessing.ipynb)
- Model training
    - [basic_cnn.ipynb](basic_cnn.ipynb)
    - [efficientnet.ipynb](efficientnet.ipynb)
    - [resnet.ipynb](resnet.ipynb)
- Model evaluation
    - [predict.ipynb](predict.ipynb)

There is also one notbook explaining the full flow: [full_flow.ipynb](full_flow.ipynb)

Further, functions and classes are available in the utils folder. 


