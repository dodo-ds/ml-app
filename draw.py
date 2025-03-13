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
        background_color="#FFFFFF",
        display_toolbar=True,
        height=252,
        width=252,
        fill_color="#000000",
    )
    run = st.button("Prediktera", use_container_width=True)


if run and canvas_result.image_data is not None:
    gray = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
    white_pixel_ratio = np.sum(gray == 255) / gray.size
    if white_pixel_ratio > 0.99:
        st.stop()
    gray = 255 - gray
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    gray = gray[y : y + h, x : x + w]
    padding = int(0.34 * h)
    gray = np.pad(gray, pad_width=((padding, padding), (padding, padding)), constant_values=0)
    processed_img = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    pred = st.session_state.model.predict([processed_img.ravel()])[0]

    st.image(gray)
    st.subheader(f"Prediktion: {pred}")

    if hasattr(st.session_state.model, "predict_proba"):
        proba = st.session_state.model.predict_proba([processed_img.ravel()])
        plt.style.use("ggplot")
        fig, ax = plt.subplots()

        ax.set_yticks(np.arange(10))
        ax.barh(np.arange(10), proba[0])
        ax.set_title("Sannolikheter")
        st.pyplot(fig)
