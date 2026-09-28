"""
НИР: Веб-приложение для распознавания объектов на изображениях
Технологии: Streamlit (интерфейс) + OpenCV DNN + MobileNet-SSD (предобученная модель)

Запуск:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

st.set_page_config(
    page_title="Распознавание объектов | НИР",
    page_icon="🔍",
    layout="wide"
)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")

BASE_DIR = os.path.dirname(__file__)
PROTOTXT = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.prototxt")
MODEL = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.caffemodel")


@st.cache_resource
def load_model():
    """Модель загружается один раз и кэшируется между запросами."""
    return cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)


def detect_objects(net, image_bgr, confidence_threshold):
    (h, w) = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)),
        scalefactor=0.007843, size=(300, 300), mean=127.5
    )
    net.setInput(blob)

    start = time.time()
    detections = net.forward()
    elapsed = time.time() - start

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label_name = CLASSES[class_id]
            color = [int(c) for c in COLORS[class_id]]

            cv2.rectangle(image_bgr, (startX, startY), (endX, endY), color, 2)
            label = f"{label_name}: {confidence*100:.1f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 15
            cv2.putText(image_bgr, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            results.append({"class": label_name, "confidence": float(confidence)})

    return image_bgr, results, elapsed


st.title("🔍 Система распознавания объектов на изображениях")
st.caption("НИР · OpenCV DNN + предобученная модель MobileNet-SSD (PASCAL VOC, 20 классов)")

with st.sidebar:
    st.header("⚙️ Параметры")
    confidence_threshold = st.slider(
        "Порог уверенности", min_value=0.1, max_value=0.9, value=0.4, step=0.05
    )
    st.markdown("---")
    st.markdown("**Распознаваемые классы:**")
    st.caption(", ".join(CLASSES[1:]))
    st.markdown("---")
    st.markdown("Загрузка модели через `cv2.dnn.readNetFromCaffe`. "
                "Веса не обучались заново — используется transfer learning / инференс на готовой модели.")

net = load_model()

uploaded_file = st.file_uploader(
    "Загрузите изображение", type=["jpg", "jpeg", "png"]
)

example_dir = os.path.join(BASE_DIR, "images")
examples = [f for f in os.listdir(example_dir)] if os.path.isdir(example_dir) else []
example_choice = None
if examples:
    example_choice = st.selectbox(
        "...или выберите готовый пример", ["— не выбрано —"] + examples
    )

image_pil = None
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
elif example_choice and example_choice != "— не выбрано —":
    image_pil = Image.open(os.path.join(example_dir, example_choice)).convert("RGB")

if image_pil is not None:
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    with st.spinner("Выполняется детекция..."):
        result_bgr, detections, elapsed = detect_objects(
            net, image_bgr.copy(), confidence_threshold
        )
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Исходное изображение")
        st.image(image_pil, use_container_width=True)
    with col2:
        st.subheader("Результат детекции")
        st.image(result_rgb, use_container_width=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Найдено объектов", len(detections))
    c2.metric("Время инференса", f"{elapsed*1000:.0f} мс")
    c3.metric("Порог уверенности", f"{confidence_threshold*100:.0f}%")

    if detections:
        st.subheader("Детали")
        st.table([
            {"класс": d["class"], "уверенность": f"{d['confidence']*100:.1f}%"}
            for d in detections
        ])
    else:
        st.info("Объекты не найдены. Попробуйте снизить порог уверенности в боковой панели.")
else:
    st.info("⬆️ Загрузите изображение или выберите готовый пример, чтобы начать.")
