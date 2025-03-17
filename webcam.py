from resources import *

KERNEL_MORPH = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
KERNEL_DILATE = np.ones((3, 3), np.uint8)
BLUR = (5, 5)
FRAME_WINDOW = st.image([])
THRESH_WINDOW = st.image([])
MIN_AREA = 50
CONF_THRESHOLD = 0.7


camera = cv2.VideoCapture(0)

run = st.button("Starta/Stoppa Kameran", use_container_width=True)
can_proba = hasattr(st.session_state.model, "predict_proba") or st.error("Välj en model som kan beräkna sannolikheter.")
while run and can_proba is True:
    ret, frame = camera.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR, 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL_MORPH, iterations=1)
    thresh = cv2.dilate(thresh, KERNEL_DILATE, iterations=1)

    H, W = thresh.shape
    y1, y2 = H // 4, 3 * H // 4
    x1, x2 = W // 8, 7 * W // 8
    mask = np.zeros_like(thresh)
    mask[y1:y2, x1:x2] = 255
    thresh_center = cv2.bitwise_and(thresh, mask)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    contours, _ = cv2.findContours(thresh_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        r = max(w, h)
        y_pad = ((w - h) // 2 if w > h else 0) + r // 5
        x_pad = ((h - w) // 2 if h > w else 0) + r // 5

        roi = thresh[y : y + h, x : x + w]
        roi = cv2.copyMakeBorder(roi, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, value=0)
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        proba = st.session_state.model.predict_proba([roi.ravel()])[0]
        if max(proba) < CONF_THRESHOLD:
            continue
        prediction = np.argmax(proba)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(
            frame,
            str(prediction),
            org=(x, y - 8),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # THRESH_WINDOW.image(thresh)
    sleep(0.01)
