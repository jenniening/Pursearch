import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.utils import secure_filename
import pathlib

import io
import numpy as np
import os
import sys

from flask import Flask, jsonify, request, flash, redirect, url_for, render_template

app = Flask(__name__)
total_brands = ['Louis Vuitton', 'Chanel', 'Coach', 'Michael Kors', 'Gucci', 'Longchamp', 'Other', 'None']
brands = [total_brands[i] for i in [1,2,0,6,1,5,4,4,1,3,3,0,0,7,0,3,4,0,4]]
names = ['2.55','F57842', 'alma','handbag','boy','classic','dionysus','ggmarmont','grandshoppingtote','jetsettote','mercertotebag',
'neverfull','newwavechaintote','None','petiteboitechapeau','selmasatchel','sohodisco','speedy','sylvie']
websites = ["https://www.chanel.com/us/fashion/p/AS0874B0228194305/small-2-55-handbag-patent-calfskin-black-metal/",
"https://www.coach.com/",
"https://us.louisvuitton.com/eng-us/search/alma",
"None",
"https://www.chanel.com/us/fashion/p/A91046Y09958N5024/large-shopping-bag-calfskin-silver-tone-metal/",
"https://us.longchamp.com/products/le-pliage/tote-bag-s/l2605089545",
"https://www.gucci.com/us/en/st/capsule/dionysus-collection",
"https://www.gucci.com/us/en/st/capsule/gg-marmont-collection",
"https://www.chanel.com/us/fashion/p/A91046Y09958N5024/large-shopping-bag-calfskin-silver-tone-metal/",
"https://www.michaelkors.com/search/_/Rtt-jet+set+",
"https://www.michaelkors.com/search/_/Rtt-mercer+",
"https://us.louisvuitton.com/eng-us/search/neverfull",
"https://us.louisvuitton.com/eng-us/search/new%20wave%20chain",
"None",
"https://us.louisvuitton.com/eng-us/products/petite-boite-chapeau-monogram-nvprod670013v",
"https://www.michaelkors.com/search/_/Rtt-selma+satchel",
"https://www.gucci.com/us/en/st/newsearchpage?searchString=soho%20disco&search-cat=header-search",
"https://us.louisvuitton.com/eng-us/search/speedy",
"https://www.gucci.com/us/en/st/newsearchpage?searchString=sylvie&search-cat=header-search"
]

UPLOAD_FOLDER = '../data/example/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}
tf.gfile = tf.io.gfile
def load_detection_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model

def load_model():
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 19)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    model_name = torch.load("../models/model_name_resnet50gpu_correct_merged", map_location=torch.device('cpu'))
    model_ft.load_state_dict(model_name)
    model_ft.eval()
    return model_ft

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    #if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
    #    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
    #          output_dict['detection_masks'], output_dict['detection_boxes'],
    #           image.shape[0], image.shape[1])      
    #    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
    #                                   tf.uint8)
    #    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict

def crop_image(image_bytes, model):
    image = Image.open(io.BytesIO(image_bytes))
    width, hight = image.size
    image_np = np.array(image)
    output_dict = run_inference_for_single_image(model, image_np)
    if 31 in output_dict["detection_classes"]:
        max_score =  max([output_dict["detection_scores"][idx] for idx, i in enumerate(output_dict["detection_classes"]) if i == 31])
        index = list(output_dict["detection_scores"]).index(max_score) 
        [y,x,h,w] = output_dict["detection_boxes"][index]
        cropped = image_np[int(y*hight):int(h*hight), int(x*width):int(w*width)]
        img = Image.fromarray(cropped,"RGB")
        return img
    else:
        return False

model = load_model()

model_name = 'faster_rcnn_inception_v2_coco_2018_01_28'
detection_model = load_detection_model(model_name)


def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return my_transforms(image).unsqueeze(0)

def get_prediction(image, model):
    tensor = transform_image(image=image)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return brands[predicted_idx], names[predicted_idx], websites[predicted_idx]

@app.route("/")
def hello():
    return render_template('home.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if not file:
            return "No handbag detected\n"

        img_bytes = file.read()
        file = crop_image(img_bytes, detection_model)
        if file == False:
            return "No handbag detected\n"
        

        brand, name, web = get_prediction(image=file, model=model)
        return render_template('result.html', class_brand=brand, class_name=name, class_web=web)
    return render_template('index.html')


#def upload_file():
#    if request.method == 'POST':
#        # check if the post request has the file part
#        if 'file' not in request.files:
#            flash('No file part')
#            return redirect(request.url)
#        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
#        if file.filename == '':
#            flash('No selected file')
#            return redirect(request.url)
        #if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
#        filename = file.filename
#        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    app.run()
    ### Command to run this: FLASK_ENV=development FLASK_APP=app.py FLASK_DEBUG=0 
