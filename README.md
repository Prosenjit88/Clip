# Clip
Find clipping in images
Steps to train the model:
•	First generate random augmented images using the script generate_images.py in the model folder.
•	After creating the augmented images, annotate the images using labelimg.
•	Next convert the annotations to tf record file using the script create_tfrecords_from_xml.py.
•	Next train the model with the file model_main.py.
•	The configuration of the model file is in file ssd_inception_v2_coco.config

The model was trained with 5000 epochs due to hardware constraint.

