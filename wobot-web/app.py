from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin


import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from tensorflow.keras.utils import load_img
# from keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array





app=Flask(__name__, template_folder="src")
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# gets the input dates from the React form and sends them to /data to get the data
@app.route('/two', methods=['GET', 'POST'])
@cross_origin()
def get_dates():

    pizza_img = request.files["pizza_data"]

    img_path = "./images/" + pizza_img.filename

    pizza_img.save(img_path)


    model = VGG16()


    image = load_img(img_path, target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)


    
    return "I did it; yes!--------------------" + classification
    
if __name__ == '__main__':
    app.run(port=5000, debug=True)