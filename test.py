from flask import Flask, request
import base64
from PIL import Image
from io import BytesIO
import numpy as np

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
        print(image)
        print(image.shape)

    return "Hello"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)