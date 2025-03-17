import streamlit as st
from resources import *


def predictor(images_uploaded: list[BytesIO]):
    image_count_container = st.empty()
    image_count = len(images_uploaded)

    columns = st.columns(4)
    col_idx = 0
    for image in images_uploaded:
        if plot_conf_matrix and image.label is None:
            st.markdown(no_label_error.format(image.name), unsafe_allow_html=True)
            image_count -= 1
            continue

        processed_img, debug_img = process_image(image)
        image.pred = st.session_state.model.predict([processed_img.ravel()])[0]

        if can_proba := hasattr(st.session_state.model, "predict_proba"):
            proba = max(st.session_state.model.predict_proba([processed_img.ravel()])[0]) * 100

        with columns[col_idx % 4]:
            st.image(debug_img, use_container_width=True)
            st.markdown(
                f"""
                <div>
                    <h5>Prediktion: {image.pred} {"✅" if image.label == image.pred else "❌" if image.label is not None else ""}</h5>
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
        {str(no_label_attached_count) + " borttagen." if no_label_attached_count else ""}
        """
    )
    return images_uploaded


def print_score(score, sidebar=True):
    emojie = "🥳" if score > 79 else "💩"
    accuracy_score_ = f"# Accuracy: {'🔥💯🔥' if score > 99.99 else f'{score:.2f} %'} {emojie}"
    if sidebar:
        st.sidebar.write(accuracy_score_)
    else:
        st.write(accuracy_score_)


def accuracy_score_(all_true_labels, all_predictions):
    if len(all_true_labels) == len(all_predictions):
        score = accuracy_score(all_true_labels, all_predictions) * 100
        return score
    else:
        st.sidebar.info(f"Antalet angivna siffror ({len(all_true_labels)}) matchar inte antalet bilder ({len(images_uploaded)})")


def conf_matrix(all_true_labels, all_predictions):
    display_confusion_matrix(all_true_labels, all_predictions, st.session_state.model.__class__.__name__)
    score = accuracy_score_(all_true_labels, all_predictions)
    print_score(score, sidebar=False)


def true_label_assigner(images_uploaded, true_labels_processed):
    if isinstance(true_labels_processed, dict):
        for image in images_uploaded:
            image.label = true_labels_processed.get(image.name)
    else:
        for image, label in zip(images_uploaded, true_labels_processed):
            image.label = label


images_uploaded = st.sidebar.file_uploader("Ladda upp bilder för prediktion...", type=["jpg", "png"], accept_multiple_files=True)

if images_uploaded:
    for image in images_uploaded:
        if not hasattr(image, "label"):
            image.label = None
        if not hasattr(image, "pred"):
            image.pred = None
else:
    st.info("Vänligen ladda upp bilder för prediktion.")

true_labels_uploaded = st.sidebar.file_uploader("Ladda upp text-filer med sanna värden...", type=["txt"], accept_multiple_files=True)
true_labels_input = st.sidebar.text_input("Eller ange sanna värden...", placeholder="Exempel: 1 3 5 7 9")

if images_uploaded and not true_labels_input and not true_labels_uploaded:
    predictor(images_uploaded)

if st.sidebar.toggle("Visa Parametrar"):
    st.sidebar.write(st.session_state.model.get_params())

if true_labels_uploaded or true_labels_input and images_uploaded:
    true_labels_processed = label_processor(true_labels_uploaded or true_labels_input)
    true_label_assigner(images_uploaded, true_labels_processed)
    predictor(images_uploaded)
    all_true_labels = [image.label for image in images_uploaded if image.label is not None]
    all_predictions = [image.pred for image in images_uploaded if image.pred is not None]
    score = accuracy_score_(all_true_labels, all_predictions)
    if score:
        plot_conf_matrix = st.sidebar.toggle("Plotta Confusion Matrix")
        print_score(score)


if plot_conf_matrix:
    conf_matrix(all_true_labels, all_predictions)
