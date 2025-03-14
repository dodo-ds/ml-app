import os
import sys
from csv import DictReader
from io import BytesIO, TextIOWrapper
from pathlib import Path
from time import sleep
from zipfile import ZipFile

import cv2
import joblib
import numpy as np
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

os.chdir(os.path.dirname(__file__))

TARGET_SIZE = (28, 28)
DEBUG_IMG_SIZE = (250, 160)
PADDING_RATIO = 0.34
KERNEL = np.ones((3, 3), np.uint8)


mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)


plot_conf_matrix = False
bg_img_url = "https://i.imgur.com/ctUckTP.png"
css = f"""
        <style>
        .stApp {{
            background-image: url("{bg_img_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        body {{
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        }}
        </style>
        """

no_label_error = """
                <div style="
                background-color: rgba(200, 0, 0, 0.4);
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;">
                <p style="color: white; margin: 0;">
                    Kunde inte hitta en etiket för bilden med filnamnet [{}].
                </p>
                </div>"""


@st.cache_resource
def load_model(path: Path):
    if path.is_file():
        if path.suffix == ".zip":
            with ZipFile(path, "r") as f:
                return joblib.load(BytesIO(f.read(f.namelist()[0])))
        else:
            return joblib.load(path)

    raise FileNotFoundError("Modellen kunde inte hittas på disk!")


@st.cache_data
def label_processor(labels: list[BytesIO] | str):
    try:
        if isinstance(labels, list):
            true_labels_processed = dict()
            for text_file in labels:
                reader = DictReader(TextIOWrapper(text_file, encoding="utf-8"))
                true_labels_processed.update({row["file"]: int(row["value"]) for row in reader})
            return true_labels_processed
        return [int(label) for label in labels.replace(" ", "").replace(",", "")]
    except Exception:
        st.sidebar.error("Ogiltigt format! Rensa och försök igen.")
        st.stop()


@st.cache_data
def process_image(uploaded_file: BytesIO):
    data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    gray_img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    user_img = cv2.imdecode(data, cv2.IMREAD_COLOR_RGB)
    _, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    white_pixel_ratio = np.sum(thresh_img == 255) / thresh_img.size
    if white_pixel_ratio > 0.5:
        thresh_img = 255 - thresh_img

    thresh_img = cv2.dilate(thresh_img, KERNEL, iterations=1)

    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        thresh_img = thresh_img[y : y + h, x : x + w]
        user_img = user_img[y : y + h, x : x + w]

        padding_size = int(max(w, h) * PADDING_RATIO)
        thresh_img = cv2.copyMakeBorder(thresh_img, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
        user_img = cv2.copyMakeBorder(user_img, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    processed_img = cv2.resize(thresh_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    debug_img = np.hstack((user_img, cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)))
    debug_img = cv2.resize(debug_img, DEBUG_IMG_SIZE, interpolation=cv2.INTER_AREA)
    return processed_img, debug_img


def display_confusion_matrix(true_labels, all_predictions, model_name=None):
    cm = confusion_matrix(true_labels, all_predictions, labels=np.arange(10))
    fig, _ = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="viridis")
    plt.title(f"Confusion Matrix för {model_name or ''}")
    plt.ylabel("Sanna värden")
    plt.xlabel("Predikterade värden")
    plt.yticks(rotation=0)
    st.pyplot(fig)
