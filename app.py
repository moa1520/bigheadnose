from tensorflow.keras.models import load_model
from flask import Flask, request
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "<h1>캡스톤 디자인</h1>"

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 데이터 수신
        json_data = request.json
        # 데이터 전처리
        image_code = base64.b64decode(str(json_data['img']))
        image = Image.open(BytesIO(image_code))
        image = np.asarray(image)

        # 이미지 사이즈 조정
        image = cv2.resize(image, (224, 224))
        #

        image = image / 255 - .5

        imgs = []
        imgs.append(image)
        imgs = np.array(imgs)

        result = model.predict(imgs)

        print("봄: " + str(round(result[0][0] * 100, 2)) + "%")
        print("여름: " + str(round(result[0][1] * 100, 2)) + "%")
        print("가을: " + str(round(result[0][2] * 100, 2)) + "%")
        print("겨울: " + str(round(result[0][3] * 100, 2)) + "%")

        res = {
            'spring': round(result[0][0] * 100, 2),
            'summer': round(result[0][1] * 100, 2),
            'fall': round(result[0][2] * 100, 2),
            'winter': round(result[0][3] * 100, 2)
        }

        json_data = json.dumps(res)
    return json_data


if __name__ == '__main__':
    model = load_model('./model/model_90.h5')
    app.run(host="0.0.0.0", port=5000)
