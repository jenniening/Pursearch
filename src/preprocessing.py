import os
import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_model(model_name):
    """
    Load the model trained using tensorflow for object detection.
    Detailed model name list can be found in 
    /Users/jianinglu1/Documents/GitHub/models/research/object_detection/g3doc/detection_model_zoo.md
    
    Parameters
    ----------
    model_name : str
        the name of model you will use to do the detection
    
    """
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

def run_inference_for_single_image(model, image):
    """
    Get the object detection results
    
    Parameters
    ----------
    model : object
        model loaded to do the object detection
    image : array
        image array used to do the object detection
    
    Returns
    -------
    output_dict: dictionary for object detection results, keys include "num_detections", "detection_classes", "detection_boxes", "detection_scores"
    """
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
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict

def crop_image(name, path, outpath, show=True, save=True):
    """
    Based on the object detection results to crop the image.
    Here, we only focus on handbag (class: 31), only keep the handbag with highest probability 
    
    Parameters
    ----------
    name : str
        image name
    path : str
        directory for original images
    outpath : str
        directory for output cropped images
    show : bool, optional
        whether show the original and cropped images, by default True
    save : bool, optional
        whether save the cropped images, by default True
    
    Returns
    -------
    True --> handbag has been founded from image
    False --> no handbag
    """
    image = Image.open(os.path.join(path,name))
    if show:
        image.show()
    width, hight = image.size
    image_np = np.array(image)
    output_dict = run_inference_for_single_image(detection_model, image_np)
    if 31 in output_dict["detection_classes"]:
        max_score =  max([output_dict["detection_scores"][idx] for idx, i in enumerate(output_dict["detection_classes"]) if i == 31])
        index = list(output_dict["detection_scores"]).index(max_score) 
        [y,x,h,w] = output_dict["detection_boxes"][index]
        cropped = image_np[int(y*hight):int(h*hight), int(x*width):int(w*width)]
        if show:
            img = Image.fromarray(cropped,"RGB")
            img.show()
        if save:
            img = Image.fromarray(cropped,"RGB")
            img.save(os.path.join(outpath, name))
        return True
    else:
        return False

def rename(path):
    """
    Rename the image by index
    
    Parameters
    ----------
    path : str
        directory for original images
    
    Returns
    -------
    The number of images in given path
    """
    olddir = os.getcwd()
    os.chdir(path)
    images = [ i for i in os.listdir(path) if i.split(".")[1] in ["png","gif","jpeg","jpg"] ]
    for idx, i in enumerate(images):
        if i.split(".")[1] in ["png","gif","jpeg"]:
            im = Image.open(i)
            rgb_im = im.convert('RGB')
            rgb_im.save(i.split(".")[0] + ".jpg")
            os.system("rm " + i )
        os.system("cp " + i.split(".")[0] + ".jpg " + str(idx) + ".jpg")
    os.chdir(olddir)
    return len(images)

def rename_otherbag(path):
    """
    Rename the image (other bags) by index
    
    Parameters
    ----------
    path : str
        directory for original images
    
    Returns
    -------
    The number of images in given path
    """
    olddir = os.getcwd()
    os.chdir(path)
    images = [ i for i in os.listdir(path) if i.split(".")[1] in ["png","gif","jpeg","jpg"] ]
    for idx, i in enumerate(images):
        if " " in i:
            os.system("cp " + i.split()[0] + "\ " + i.split()[1].split(".")[0] + ".jpg " + str(idx) + ".jpg")
        else:
            os.system("cp " + i.split(".")[0] + ".jpg " + str(idx) + ".jpg")
    os.chdir(olddir)
    return len(images)

if __name__ == "__main__":
    # patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1
    # Patch the location of gfile
    tf.gfile = tf.io.gfile  
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = '/Users/jianinglu1/Documents/GitHub/models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    model_name = 'faster_rcnn_inception_v2_coco_2018_01_28' 
    detection_model = load_model(model_name)

    for i in ["jetsettote","mercertotebag","selmasatchel"]:
        number = rename("/Users/jianinglu1/Documents/GitHub/DataInsight/data/clean/MK/" + i)
        path = "/Users/jianinglu1/Documents/GitHub/DataInsight/data/clean/MK/" + i
        outpath = "/Users/jianinglu1/Documents/GitHub/DataInsight/data/processed/MK/" + i
        for i in range(number):
            name = str(i) + ".jpg"
            try:
                if not crop_image(name, path, outpath, show=False):
                    print(i)
            except:
                print(i)




