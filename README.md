# Pursearch
Pursearch is a web app to help user find their dream handbag from unlabelled social media photos.<br>
## 1. Motivation
**Handbag** is one of the most critical component of fashion market. Everyday, millions of people and celebrities share their fashion look on social media and people can always find the handbag they are interested in from these photos. However, the detailed information of handbags such as brands and style names aren't always shared. People have to ask and wait for the response for several hours or days, which is a big obstacle for potential handbag customers since many people will loss interests during this long waiting process. Because of this, I developed Pursearch, an image based handbag searching tool, to help users identify the detailed information of handbag and find the affordable similar handbags easily.
## 2. Website (http://Pursearch.com)
### Home Page: 
<img src="https://github.com/jenniening/Pursearch/blob/master/demo/Homepage.png" width="771"></img>
### Input Page:
User can choose to crop image or not. Crop image is defualt and means that handbag will be first detected from input image. However, handbag detection might be failed, in that case, user should choose to use No Crop.<br>
User can shoose different price and retailer for handbag recommendation.<br>
<img src="https://github.com/jenniening/Pursearch/blob/master/demo/Input.png" width="771" ></img>
### Result Page:
First row shows the input image and cropped handbag image. Handbag identification results and Top3 handbags based on similarity and users' choices have been provided. <br>
<img src="https://github.com/jenniening/Pursearch/blob/master/demo/Pursearchresult.png" width="771" ></img>
## 3. Pipeline 
<img src="https://github.com/jenniening/Pursearch/blob/master/demo/Pipeline.png" width="771"></img>
### Data Source
Dataset includes two parts: branded handbag dataset and retailer handbag dataset<br> 
**Branded handbag dataset** includes 5k+ images with 6 most popular handbag brands and 18 different styles of handbags scrapped from **Google search results**. <br> **Retailer handbag dataset** includes images and other information of 101 brands and 3k+ styles of handbags scrapped from **Saks Fifth Avenue** and **Shein**. 
### Create Environments
```
make Makefile create_environment
conda activate Pursearch
make Makefile requirements
```
### Training Models 
All scripts are in ```src``` folder<br>
To preprocess the data, I used faster_rcnn model trained by COCO dataset using Tensorflow to detect handbag from original image (preprocess.py).<br>
In terms of handbag identification, I built CNN model based on pretrained ResNet_50 and retrain all weights using branded handbag dataset and PyTorch. <br> To train the model:
```
python train.py
```
### Test Application
Application can be run locally by running following command: 
```
python app.py
```
## 4. Tutorials
[Handbag Detection](https://github.com/jenniening/Pursearch/blob/master/tutorials/Handbag_Detection.ipynb)<br>
[Handbag Identification](https://github.com/jenniening/Pursearch/blob/master/tutorials/Handbag_Classification.ipynb)<br>
[Handbag Recommendation](https://github.com/jenniening/Pursearch/blob/master/tutorials/Handbag_Recommendation.ipynb)<br>



