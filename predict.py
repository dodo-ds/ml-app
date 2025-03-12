import streamlit as st
from resources import *


def predictor(images_uploaded: list[BytesIO], true_labels_processed: dict):
    image_count_container = st.empty()
    image_count = len(images_uploaded)
    all_predictions = []
    all_true_labels = []
    columns = st.columns(4)
    col_idx = 0
    for image in images_uploaded:
        if true_labels_toggled:
            if (label := true_labels_processed.get(image.name)) is None:
                # st.error(f"Kunde inte hitta en etiket för bilden med filnamnet [{image.name}].")
                st.markdown(no_label_error.format(image.name), unsafe_allow_html=True)
                image_count -= 1
                continue
            all_true_labels.append(label)

        processed_img, debug_img = process_image(image)
        pred = st.session_state.model.predict([processed_img.ravel()])[0]

        if can_proba := hasattr(st.session_state.model, "predict_proba"):
            proba = max(st.session_state.model.predict_proba([processed_img.ravel()])[0]) * 100

        all_predictions.append(pred)

        with columns[col_idx % 4]:
            st.image(debug_img)
            st.markdown(
                f"""
                <div>
                    <h5>Prediktion: {pred} {"✅" if true_labels_toggled and label == pred else "❌" if true_labels_toggled and label != pred else ""}</h5>
                    {"<p>Sannolikhet: " + str(round(proba, 2)) + "%</p>" if can_proba else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )
        col_idx += 1
    no_label_attached_count = len(images_uploaded) - image_count
    image_count_container.write(
        f"""
        {image_count}{" instanser predikerade." if image_count > 1 else " instans predikerad."} \
        {str(no_label_attached_count) + " borttagen." if true_labels_toggled and no_label_attached_count else ""}
        """
    )
    return all_true_labels, all_predictions


def conf_matrix(all_true_labels, all_predictions):
    try:
        true_labels = all_true_labels if true_labels_toggled else [int(label) for label in true_labels_input.replace(" ", "").replace(",", "")]
    except ValueError:
        st.sidebar.error("Ogiltigt format! Använd endast siffror.")
    else:
        if len(true_labels) == len(all_predictions):
            score = accuracy_score(true_labels, all_predictions) * 100
            display_confusion_matrix(true_labels, all_predictions, st.session_state.model.__class__.__name__)
            emojie = "🥳" if score > 79 else "💩"
            accuracy_score_ = f"### Accuracy: {'🔥💯🔥' if score > 99.99 else f'{score:.2f} %'} {emojie}"
            st.write(accuracy_score_)
            st.sidebar.write(accuracy_score_)
        else:
            st.sidebar.error(f"Antalet angivna siffror ({len(true_labels)}) matchar inte antalet bilder ({len(images_uploaded)})!")


images_uploaded = st.sidebar.file_uploader("Ladda upp bilder för prediktion...", type=["jpg", "png"], accept_multiple_files=True)
true_labels_uploaded = st.sidebar.file_uploader("Ladda upp text-filer med sanna värden...", type=["txt"], accept_multiple_files=True)

if true_labels_uploaded:
    true_labels_processed = label_processor(true_labels_uploaded)

true_labels_input = st.sidebar.text_input("Eller ange sanna värden...", placeholder="Exempel: 1 3 5 7 9")

if true_labels_uploaded:
    true_labels_toggled = st.sidebar.toggle("Använd text-filer som sanna värden")
model_chosen = st.sidebar.radio("Välj Model:", st.session_state.models)

if model_chosen:
    if isinstance(model_chosen, Path):
        st.session_state.model: ExtraTreesClassifier | SVC = load_model(model_chosen)  # type: ignore
    else:
        st.session_state.model: ExtraTreesClassifier | SVC = model_chosen  # type: ignore

if images_uploaded:
    all_true_labels, all_predictions = predictor(images_uploaded, true_labels_processed if true_labels_uploaded else {})
else:
    st.info("Vänligen ladda upp bilder för prediktion.")


st.sidebar.write("Aktiv:", st.session_state.model.__class__.__name__)


if st.sidebar.toggle("Visa Parametrar"):
    st.sidebar.write(st.session_state.model.get_params())


if images_uploaded and model_chosen and (true_labels_toggled or true_labels_input):
    conf_matrix(all_true_labels, all_predictions)
