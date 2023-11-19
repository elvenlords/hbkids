# this program demonstrates the use of a pre-trained
# imagenet model in order to find out what is there
# in an  image. This program is very naive in that it
# reads one image and then asks the model to make a 
# prediction. Once the model returns a prediction
# it just decodes the first 4 predictions along with
# the probability values for each of the prediction

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np

# Load MobileNetV2 model pre-trained on ImageNet data
model = MobileNetV2(weights='imagenet')

# Load an image for classification
img_path = 'tm2.jpg'  # Replace with the path to your image file
img = image.load_img(img_path, target_size=(224, 224))  # MobileNetV2 input size

# Convert the image to a numpy array and preprocess it
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)

# Decode and print the top-3 predicted classes
decoded_predictions = decode_predictions(predictions, top=5)[0]
print("Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
