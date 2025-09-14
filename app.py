import streamlit as st
from ultralytics import YOLO
import torch
from torchvision import models, transforms
from PIL import Image
import io

# --- Конфигурация ---
DETECTOR_WEIGHTS_PATH = 'weights/best.pt'
CLASSIFIER_WEIGHTS_PATH = 'weights/cleanliness_classifier_best.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Загрузка моделей (с кэшированием для скорости) ---
@st.cache_resource
def load_detector():
    model = YOLO(DETECTOR_WEIGHTS_PATH)
    return model

@st.cache_resource
def load_classifier():
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS_PATH, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval()
    return model

# --- Функция для предсказания ---
def predict(image_bytes):
    # 1. Предсказание чистоты
    classifier_model = load_classifier()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_for_classifier = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = transform(image_for_classifier)
    batch_t = torch.unsqueeze(img_t, 0).to(DEVICE)
    with torch.no_grad():
        output = classifier_model(batch_t)
        prob = torch.sigmoid(output).item()
    
    cleanliness_result = "Чистый" if prob < 0.5 else "Грязный" # Инвертируйте, если классы перепутаны
    cleanliness_confidence = 1 - prob if prob < 0.5 else prob
    
    # 2. Предсказание повреждений
    detector_model = load_detector()
    image_for_detector = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    detection_results = detector_model(image_for_detector)[0]
    
    # 3. Визуализация
    annotated_image = detection_results.plot() # YOLOv8 имеет удобный метод plot
    annotated_image = Image.fromarray(annotated_image[..., ::-1]) # BGR -> RGB

    num_damages = len(detection_results.boxes)
    integrity_result = "Целый" if num_damages == 0 else "Есть повреждения"
    
    return cleanliness_result, cleanliness_confidence, integrity_result, num_damages, annotated_image

# --- Интерфейс Streamlit ---
st.set_page_config(layout="wide", page_title="Анализатор состояния автомобиля")
st.title("🚗 Система определения состояния автомобиля")
st.write("Прототип для хакатона inDrive. Загрузите фотографию автомобиля для анализа.")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Оригинальное изображение")
        st.image(image_bytes, use_column_width=True)

    with col2:
        st.subheader("Результаты анализа")
        with st.spinner('Анализирую...'):
            clean_res, clean_conf, int_res, int_num, ann_img = predict(image_bytes)
            
            st.metric(label="Чистота", value=clean_res)
            st.metric(label="Целостность", value=int_res, delta=f"-{int_num} дефектов найдено" if int_num > 0 else "Дефектов не найдено")

            st.image(ann_img, caption="Обнаруженные повреждения", use_column_width=True)