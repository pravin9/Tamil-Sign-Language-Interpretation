from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import logging
from werkzeug.utils import secure_filename
from test_deploy import sign_detection, sentence_prediction
from capture_images import conversion, getresult, getvideoframes
import math
import os
from flask import Flask, render_template, request
import shutil

app = Flask(__name__)

file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

TAMIL_SIGN_LANGUAGE_HOME = os.path.dirname(os.path.realpath(__file__))
SIGN_IMAGE_UPLOAD_FOLDER = 'C:/Users/User/Downloads/Flask/uploads'
app.config['SIGN_IMAGE_UPLOAD_FOLDER'] = SIGN_IMAGE_UPLOAD_FOLDER

sentences = ["சைகை மொழி உங்களால் புரிந்துகொள்ள முடியுமா", 
       "எனக்கு தமிழ் கற்றுத்தர முடியுமா", 
       "நான் உன்னை காதலிக்கிறேன்",
       "உனக்கு தமிழ் புரியுமா",
       "நான் என் வேலையை முடிக்க வேண்டும்",
       "நான் தமிழ் படித்து முடிக்க வேண்டும்",
       "எனக்கு படிக்க உதவ முடியுமா",
       "நான் உங்களுக்கு நல்ல சைகை மொழி கற்றுத்தர கொடுக்க முடியும்",
       "மொழிபெயர்ப்பாளர் எனக்கு தமிழ் படிக்க உதவுகிறார்",
       "எனக்கு விளையாடவும் படிக்கவும் பிடிக்கும்",
       "நீங்கள் வேலையை முடித்துவிட்டீர்களா",
       "நீங்கள் எனக்கு வேலையை முடிக்க உதவ முடியுமா",
       "சைகை மொழி புரிந்துகொள்ள மொழிபெயர்ப்பாளர் உங்களுக்கு உதவ முடியும்",
       "மன்னிக்கவும் நான் தமிழ் படிக்க வேண்டும்",
       "நான் சைகை மொழி கற்றுத்தர விரும்புகிறேன்"]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        shutil.rmtree(app.config['SIGN_IMAGE_UPLOAD_FOLDER'])
        os.mkdir(app.config['SIGN_IMAGE_UPLOAD_FOLDER'])
        uploaded_image = request.files.getlist('image')
        for file in uploaded_image:
            uploaded_file_type = str(file.filename).split(".")[-1]
            print(f"The uploaded_file_type is {uploaded_file_type}")
            uploaded_img_name = secure_filename(file.filename)
            saved_path = os.path.join(app.config['SIGN_IMAGE_UPLOAD_FOLDER'], uploaded_img_name)
            file.save(saved_path)
            if uploaded_file_type == "mp4":
                getvideoframes(saved_path)
        path, dirs, files = next(os.walk(app.config['SIGN_IMAGE_UPLOAD_FOLDER']))
        file_count = len(files)
        tamil_signs = sign_detection(app.config['SIGN_IMAGE_UPLOAD_FOLDER'], file_count)
        tamil_signs = list(dict.fromkeys(tamil_signs))
        if not tamil_signs:
            no_sign = "NO SIGN DETECTED"
            return render_template('tamil_sign_language.html', output=no_sign)
        else:
            my_dict = conversion()
            result = getresult(my_dict, tamil_signs)
            tag = sentence_prediction(result)
            output = sentences[tag]
            return render_template('tamil_sign_language.html', output=output)
    return render_template('tamil_sign_language.html')



class Config:

    def __init__(self):
        self.verbose = True

        self.network = 'vgg'

        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False

        self.anchor_box_scales = [64, 128, 256]

        self.anchor_box_ratios = [
            [1, 1], [1. / math.sqrt(2), 2. / math.sqrt(2)], [2. / math.sqrt(2), 1. / math.sqrt(2)]]

        self.im_size = 300

        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        self.num_rois = 4

        self.rpn_stride = 16

        self.balanced_classes = False

        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        self.class_mapping = None

        self.model_path = None


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='localhost', port=port)
