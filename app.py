from pathlib import Path

import streamlit as st
from resources import css

if "models" not in st.session_state:
    st.session_state.models = [model for model in Path("resources/models").glob("*.*")]

st.set_page_config(page_title="DS24 ML", page_icon=":1234:")
st.title("DS24 Machine Learning")

st.markdown(css, unsafe_allow_html=True)


pages = [
    st.Page("predict.py", title="1. Ladda upp bilder och prediktera"),
    st.Page("draw.py", title="2. Rita och prediktera"),
    st.Page("train_model.py", title="3. Träna modeller"),
]

pg = st.navigation(pages, expanded=True)

pg.run()
