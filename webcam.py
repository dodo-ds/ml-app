from resources import *

KERNEL_MORPH = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
KERNEL_DILATE = np.ones((3, 3), np.uint8)
FRAME_WINDOW = st.image([])
THRESH_WINDOW = st.image([])

run = st.button("Starta/Stoppa Kameran", use_container_width=True)
camera = cv2.VideoCapture(0)
while run:
    ret, frame = camera.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL_MORPH, iterations=1)
    thresh = cv2.dilate(thresh, KERNEL_DILATE, iterations=1)

    H, W = thresh.shape
    y1, y2 = H // 4, 3 * H // 4
    x1, x2 = W // 8, 7 * W // 8
    mask = np.zeros_like(thresh)
    mask[y1:y2, x1:x2] = 255
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    thresh_center = cv2.bitwise_and(thresh, mask)

    contours, _ = cv2.findContours(thresh_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        min_area, max_area = 50, 700
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            roi = thresh[y : y + h, x : x + w]
            padding_size = int(max(w, h) * 0.34)
            roi = cv2.copyMakeBorder(roi, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            if hasattr(st.session_state.model, "predict_proba"):
                if max(st.session_state.model.predict_proba([roi.ravel()])[0]) < 0.55:
                    continue
            prediction = st.session_state.model.predict([roi.ravel()])[0]
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
    sleep(0.01)
    # THRESH_WINDOW.image(thresh)
