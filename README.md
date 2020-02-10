# Pursearch
Pursearch is a web app to help user find their dream handbag from unlabelled social media photos.<br>
## 1. Motivation
**Handbag** is one of the most critical component of fashion market. Everyday, millions of people and celebrities share their fashion look on social media and people can always find the handbag they are interested in. However, the detailed information of handbag such as brand and style name aren't always shared. People have to ask and wait for the response for several hours or days, which is a big obstacle for potential handbag customers since many people will loss interests during this long waiting process. Because of this, I developed Pursearch, an image based handbag searching tool, to help users identify the detailed information of handbag and find the affordable similar handbags easily.
## 1. Demo
The demo of Pursearch can be found in [Demo](http://Pursearch.com)<br>
### Home Page: 
<img src="https://github.com/jenniening/Pursearch/blob/master/demo/Homepage.png" width="771"></img>
### Input Page:
User can choose to crop image or not. Crop image is defualt and means that handbag will be first detected from input image. However, handbag detection might be failed, in that case, user should choose to use No Crop.<br>
User can shoose different price and retailer for handbag recommendation.<br>
<img src="https://github.com/jenniening/Pursearch/blob/master/demo/Input.png" width="771" ></img>
### Result Page:
First row shows the input image and cropped handbag image. Handbag identification results and Top3 handbags based on similarity and users' choices have been provided. <br>
<img src="https://github.com/jenniening/Pursearch/blob/master/demo/Pursearchresult.png" width="771" ></img>
## 2. Pipeline 
<img src="https://github.com/jenniening/Pursearch/blob/master/demo/Pipeline.png" width="771"></img>
### Dataset Prepartion
Dataset includes two parts: branded handbag dataset and retailer handbag dataset<br>
#### Branded Handbag Dataset:
5k+ images include 6 most popular handbag brands and 18 different styles of handbags have been scraped from Google search results.
#### Retailer Handbag Dataset:
The images and other information of 101 brands and 3k+ styles of handbags from Saks Fifth Avenue and Shein have been scraped. 
### Handbag Detection
Here, I used faster_rcnn model trained by COCO dataset using Tensorflow to detect handbag from original image.
### Handbag Identification
I built CNN model based on pretrained ResNet_50 and retrain all weights using branded handbag dataset and PyTorch. The vector of fully connected layer has been used as handbag vector.
### Handbag Recommendation
Cosine similarity between input handbag vector and retailer handbag vectors have been calculated. Based on user's price choice and retailer choice, similar handbag will be recommended. 

