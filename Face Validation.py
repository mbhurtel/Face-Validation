import cv2
import dlib
from PIL import Image
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.backend as K

# Loading the input image
input_image = "test images/test1.jpg"

img = cv2.imread(input_image)

# Initializing the frontal face detector object
face_detector = dlib.get_frontal_face_detector()

# Detecting the faces
faces = face_detector(img)
	
# Getting the number of faces	
numberOfFaces = len(faces)

output_classes = {0: "Masked", 1: "Unmasked"}
def maskedUnmasked():

	# Loading the model to check masked or unmasked
	model = load_model('modelMask.hdf5', compile=False)

	# Loading the image and resizing to (64, 64) resolution
	img = image.load_img(input_image, target_size=(64, 64))

	# Converting to array
	imgArr = image.img_to_array(img)

	# Exapnding the dimension of the array to make compatible to predict method
	imgArr = np.expand_dims(imgArr, axis=0)

	# Predicting the category and flattening the result to lower dimension
	pred = model.predict(imgArr).flatten()

	# Extracting the index of the maximum softmax value
	pred_class = np.argmax(pred)

	# Finally getting the predicted category
	predicted_category = output_classes[pred_class]

	return predicted_category

# Checking the forntal face detected
if faces and (numberOfFaces == 1) and (maskedUnmasked() == "Masked"):
    print("Success")

else:
    print("Please insert a valid photo with the criteria:\n")
    print("1. Front Facing Human Photo \n2. Single Faced Photo \n3. Person not wearing mask")