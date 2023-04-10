from flask import Flask, request, render_template
import cv2
import numpy as np
import keras.backend as K
from keras.models import load_model

app = Flask(__name__)

# Load the character list used for training
char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Load the ML model
act_model = load_model('model.h5')

# Function to preprocess the image
def process_image(img):
    w, h = img.shape

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    # Normalize 
    img = img / 255

    return img

# Function to encode the output word into digits
def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, chara in enumerate(txt):
        dig_lst.append(char_list.index(chara))

    return dig_lst

# Route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the image upload
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        img = request.files['image'].read()
        npimg = np.fromstring(img, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image
        img = process_image(img)

        # Make the prediction
        prediction = act_model.predict(np.expand_dims(img, axis=0))

        # Decode the prediction
        out = K.get_value(K.ctc_decode(prediction, 
                            input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                            greedy=True)[0][0])

        mask = out != -1

        new_out = out[mask]

        # Convert the output digits to characters
        out = ''.join([char_list[char] for char in new_out])

        # Render the result
        return render_template('result.html', result=out)

if __name__ == '__main__':
    app.run(debug=True)