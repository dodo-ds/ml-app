from pathlib import Path

import streamlit as st
from resources import css, load_model, radio

if "models" not in st.session_state:
    st.session_state.models = [model for model in Path("resources/models").glob("*.*")]


st.set_page_config(page_title="DS24 ML", page_icon=":1234:")
st.title("DS24 Machine Learning")

st.markdown(css, unsafe_allow_html=True)

st.markdown(radio, unsafe_allow_html=True)

model_chosen = st.sidebar.radio("Välj Model:", st.session_state.models)

if model_chosen:
    if isinstance(model_chosen, Path):
        st.session_state.model = load_model(model_chosen)
    else:
        st.session_state.model = model_chosen

st.sidebar.write("Aktiv:", st.session_state.model.__class__.__name__)

pages = [
    st.Page("predict.py", title="1. Ladda upp bilder och prediktera"),
    st.Page("draw.py", title="2. Rita och prediktera"),
    st.Page("webcam.py", title="3. Öppna kameran och prediktera i realtid"),
    st.Page("train_model.py", title="4. Träna modeller"),
]

pg = st.navigation(pages, expanded=True)

pg.run()
