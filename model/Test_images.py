from matplotlib import pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import matplotlib
from PIL import Image
import matplotlib.patches as patches
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse
import sys
import cv2

# example python test_model.py frozen_inference_graph.pb images/1_1.jpg


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-m', "--model", required=True,
	help="path to the model")
ap.add_argument('-t', "--image", required=True,
	help="path to the test image")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to modefied images")
ap.add_argument("-l", "--label", required=True,
	help="path to label map file")
args = vars(ap.parse_args())


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args["model"], 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(args["label"])
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    image_np = cv2.imread(args["image"])
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

    if scores[0][0] < 0.1:
        sys.exit('Clip not found')

    print('Clip found')
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    cv2.imwrite(args["output"],image_np)
    
