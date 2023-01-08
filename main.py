import numpy as np
import pandas as pd
import cv2
from keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

@st.cache(allow_output_mutation=True)
def load_keras_model(path):
    return load_model(path, compile=False)

def transform(image):
    image = image[:, :, 0].astype('uint8') # 0番目のチャンネルだけ取り出して，白黒画像にする
    image = cv2.resize(image, dsize=(28, 28))
    image = image.astype('float32')
    image /= 255
    return image.reshape(1, 28, 28, 1)

def predict(model, data):
    if np.all(data == data[0, 0, 0, 0]): # 初期画像ならすべての要素が0の配列を返す
        return np.zeros(10)
    p = model.predict(data)[0]
    return p

model = load_keras_model("model.h5")

st.title("digit-recognition-app")

col1, col2 = st.columns(2)
with col1:
    st.caption("0~9までの数字を記入してください")
    canvas_result = st_canvas(
        stroke_width=15,
        stroke_color="#fff",
        background_color="#000",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
with col2:
    if canvas_result.image_data is not None:
        pred = predict(model, transform(canvas_result.image_data))

        st.caption("予測: {}".format(pred.argmax()))
        chart_data = pd.DataFrame(
            pred,
            columns=["predict"]
        )
        st.bar_chart(chart_data, use_container_width=True)

st.write("Code: https://github.com/s-usukura/digit-recognition-app")
