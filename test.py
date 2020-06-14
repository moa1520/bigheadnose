from flask import Flask, request
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import json

app = Flask(__name__)


@app.route("/")
def hello():
    return "hello"


@app.route("/test", methods=['POST'])
def test():
    if request.method == 'POST':
        # 데이터 수신
        json_data = request.json
        # 데이터 전처리
        image_code = base64.b64decode(str(json_data['img']))
        image = Image.open(BytesIO(image_code))
        image = np.asarray(image)
        print(image.shape)

        res = {
            'spring': 0,
            'summer': 35,
            'fall': 10,
            'winter': 55
        }

        json_data = json.dumps(res)

    return json_data


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)