from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from resources import load_model
from streamlit_drawable_canvas import st_canvas

model_chosen = st.sidebar.radio("Välj Model:", st.session_state.models)

if model_chosen:
    if isinstance(model_chosen, Path):
        st.session_state.model = load_model(model_chosen)
    else:
        st.session_state.model = model_chosen

st.sidebar.write("Aktiv:", st.session_state.model.__class__.__name__)

col, _ = st.columns([20, 34])
with col:
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="#000000",
        background_color="#eee",
        display_toolbar=True,
        height=252,
        width=252,
        fill_color="#000000",
    )
    run = st.button("Prediktera", use_container_width=True)


if run and canvas_result.image_data is not None:
    gray = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
    _, tresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if np.sum(tresh_img == 255) / tresh_img.size > 0.5:
        st.rerun()

    contours, _ = cv2.findContours(tresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    tresh_img = tresh_img[y : y + h, x : x + w]
    gray = gray[y : y + h, x : x + w]
    padding = int(0.24 * h)

    tresh_img = np.pad(tresh_img, pad_width=((padding, padding), (padding, padding)), constant_values=0)
    gray = np.pad(gray, pad_width=((padding, padding), (padding, padding)), constant_values=255)

    processed_img = cv2.resize(tresh_img, (28, 28), interpolation=cv2.INTER_AREA)

    pred = st.session_state.model.predict([processed_img.ravel()])[0]

    debug_img = np.hstack((gray, tresh_img))
    st.image(tresh_img)
    st.subheader(f"Prediktion: {pred}")

    if hasattr(st.session_state.model, "predict_proba"):
        proba = st.session_state.model.predict_proba([processed_img.ravel()])
        plt.style.use("ggplot")
        fig, ax = plt.subplots()

        ax.set_yticks(np.arange(10))
        ax.barh(np.arange(10), proba[0])
        ax.set_title("Sannolikheter")
        st.pyplot(fig)
