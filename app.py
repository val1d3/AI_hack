import streamlit as st
import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO
from supervision import LineZone, BoxAnnotator, LineZoneAnnotator, Detections
from supervision.geometry.core import Point
from datetime import datetime, timedelta
import tempfile
import numpy as np
from collections import OrderedDict
import os

# --- Настройки ---
st.title("🚉 Анализ видео с железнодорожной платформы")
st.sidebar.header("Настройки")

confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.4)
line_y = st.sidebar.slider("Позиция линии", 0, 1080, 600)
skip_frames = st.sidebar.slider("Пропускать кадров", 1, 10, 2)
resize_factor = st.sidebar.slider("Уменьшение разрешения", 0.3, 1.0, 0.6)
disable_ocr = st.sidebar.checkbox("Отключить OCR (распознавание времени)", value=False)

# Настройки области OCR
ocr_x = st.sidebar.slider("X-позиция OCR", 0, 1000, 10)
ocr_y = st.sidebar.slider("Y-позиция OCR", 0, 1000, 10)
ocr_width = st.sidebar.slider("Ширина OCR", 100, 800, 300)
ocr_height = st.sidebar.slider("Высота OCR", 20, 200, 50)

# Настройки зоны детекции поезда
st.sidebar.subheader("Зона детекции поезда")
train_zone_x = st.sidebar.slider("Начало зоны поезда (X)", 0, 1000, 500)
train_zone_width = st.sidebar.slider("Ширина зоны поезда", 100, 1000, 500)
train_zone_y = st.sidebar.slider("Начало зоны поезда (Y)", 0, 800, 200)
train_zone_height = st.sidebar.slider("Высота зоны поезда", 100, 1000, 400)

# Настройки цветовой фильтрации поезда
st.sidebar.subheader("Цветовая фильтрация поезда")
gray_lower = st.sidebar.slider("Серый нижний порог", 0, 255, 50)
gray_upper = st.sidebar.slider("Серый верхний порог", 0, 255, 200)
orange_lower_h = st.sidebar.slider("Оранжевый H нижний", 0, 180, 5)
orange_upper_h = st.sidebar.slider("Оранжевый H верхний", 0, 180, 15)
red_lower_h1 = st.sidebar.slider("Красный H1 нижний", 0, 180, 0)
red_upper_h1 = st.sidebar.slider("Красный H1 верхний", 0, 180, 10)
red_lower_h2 = st.sidebar.slider("Красный H2 нижний", 0, 180, 170)
red_upper_h2 = st.sidebar.slider("Красный H2 верхний", 0, 180, 180)

# ✅ УПРОЩЕННЫЙ ReID на основе OpenCV
st.sidebar.subheader("Настройки ReID")
reid_threshold = st.sidebar.slider("Порог ReID сходства", 0.1, 1.0, 0.6)
enable_reid = st.sidebar.checkbox("Включить ReID", value=True)

# Модель
model = YOLO("yolov8n.pt")

# OCR ридер
reader = None
if not disable_ocr:
    try:
        reader = easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        st.warning(f"Не удалось инициализировать OCR: {e}")
        reader = None

# Аннотаторы
box_annotator = BoxAnnotator()
line_annotator = LineZoneAnnotator()
line = LineZone(start=Point(0, line_y), end=Point(9999, line_y))

# Хранилища данных
people_data = []
train_events = []
occupancy = []

# ✅ УПРОЩЕННЫЙ ReID на основе гистограмм и ORB features
class SimpleReIDStorage:
    def __init__(self, similarity_threshold=0.6):
        self.similarity_threshold = similarity_threshold
        self.known_descriptors = OrderedDict()  # ID -> ORB descriptors
        self.known_histograms = OrderedDict()   # ID -> цветовые гистограммы
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def extract_features(self, image, bbox):
        """Извлекает особенности из изображения человека"""
        try:
            # Вырезаем область человека
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            person_crop = image[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return None, None
                
            # Увеличиваем контраст для лучшего извлечения признаков
            lab = cv2.cvtColor(person_crop, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Извлекаем ORB дескрипторы
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            # Вычисляем цветовые гистограммы
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
            
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            histogram = np.concatenate([hist_h, hist_s, hist_v])
            
            return descriptors, histogram
        except Exception as e:
            return None, None
    
    def calculate_similarity(self, desc1, hist1, desc2, hist2):
        """Вычисляет схожесть между двумя наборами признаков"""
        similarity = 0.0
        
        # Сравниваем гистограммы (70% веса)
        if hist1 is not None and hist2 is not None:
            hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            similarity += hist_sim * 0.7
        
        # Сравниваем ORB дескрипторы (30% веса)
        if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
            try:
                matches = self.bf.match(desc1, desc2)
                if len(matches) > 0:
                    orb_sim = len(matches) / min(len(desc1), len(desc2))
                    similarity += min(orb_sim, 1.0) * 0.3
            except:
                pass
        
        return similarity
    
    def find_best_match(self, new_descriptors, new_histogram):
        """Находит лучшего кандидата для повторной идентификации"""
        if not self.known_descriptors:
            return None, 0.0
            
        best_match_id = None
        best_similarity = 0.0
        
        for person_id in self.known_descriptors:
            stored_desc, stored_hist = self.known_descriptors[person_id], self.known_histograms[person_id]
            similarity = self.calculate_similarity(new_descriptors, new_histogram, stored_desc, stored_hist)
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = person_id
                
        return best_match_id, best_similarity
    
    def add_person(self, person_id, descriptors, histogram):
        """Добавляет нового человека в хранилище"""
        self.known_descriptors[person_id] = descriptors
        self.known_histograms[person_id] = histogram
    
    def update_person(self, person_id, descriptors, histogram):
        """Обновляет данные существующего человека"""
        if person_id in self.known_descriptors:
            # Обновляем дескрипторы (сохраняем лучшие)
            old_desc = self.known_descriptors[person_id]
            if descriptors is not None and len(descriptors) > len(old_desc):
                self.known_descriptors[person_id] = descriptors
            
            # Обновляем гистограмму (скользящее среднее)
            old_hist = self.known_histograms[person_id]
            alpha = 0.7
            new_hist = alpha * old_hist + (1 - alpha) * histogram
            self.known_histograms[person_id] = new_hist
        else:
            self.add_person(person_id, descriptors, histogram)

# Инициализация упрощенного ReID
reid_storage = SimpleReIDStorage(similarity_threshold=reid_threshold) if enable_reid else None

# Функции для цветовой фильтрации
def detect_train_colors(roi):
    """Определяет наличие цветов поезда в области"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    gray_mask = cv2.inRange(roi, (gray_lower, gray_lower, gray_lower), (gray_upper, gray_upper, gray_upper))
    
    lower_orange = np.array([orange_lower_h, 100, 100])
    upper_orange = np.array([orange_upper_h, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    lower_red1 = np.array([red_lower_h1, 100, 100])
    upper_red1 = np.array([red_upper_h1, 255, 255])
    lower_red2 = np.array([red_lower_h2, 100, 100])
    upper_red2 = np.array([red_upper_h2, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    combined_mask = cv2.bitwise_or(gray_mask, orange_mask)
    combined_mask = cv2.bitwise_or(combined_mask, red_mask)
    
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return 0, 0, 0, 0
    
    gray_percent = np.sum(gray_mask > 0) / total_pixels * 100
    orange_percent = np.sum(orange_mask > 0) / total_pixels * 100
    red_percent = np.sum(red_mask > 0) / total_pixels * 100
    combined_percent = np.sum(combined_mask > 0) / total_pixels * 100
    
    return gray_percent, orange_percent, red_percent, combined_percent

def is_in_train_zone(x1, y1, x2, y2, frame_width, frame_height):
    """Проверяет, находится ли объект в зоне детекции поезда"""
    zone_x1 = train_zone_x
    zone_x2 = min(frame_width, train_zone_x + train_zone_width)
    zone_y1 = train_zone_y
    zone_y2 = min(frame_height, train_zone_y + train_zone_height)
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    in_zone = (zone_x1 <= center_x <= zone_x2) and (zone_y1 <= center_y <= zone_y2)
    
    overlap_x = max(0, min(x2, zone_x2) - max(x1, zone_x1))
    overlap_y = max(0, min(y2, zone_y2) - max(y1, zone_y1))
    overlap_area = overlap_x * overlap_y
    object_area = (x2 - x1) * (y2 - y1)
    
    if object_area > 0:
        overlap_ratio = overlap_area / object_area
    else:
        overlap_ratio = 0
    
    return in_zone or overlap_ratio > 0.3, (zone_x1, zone_y1, zone_x2, zone_y2)

uploaded_file = st.file_uploader("Загрузи видео с платформы", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    process_w = int(original_w * resize_factor)
    process_h = int(original_h * resize_factor)

    stframe = st.empty()
    progress = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Для поезда
    train_present = False
    train_arrival_time = None
    tracked_ids = set()
    
    # Словарь для маппинга старых ID к новым через ReID
    id_mapping = {}  # новый_tracker_id -> старый_person_id
    next_person_id = 1
    
    # Счетчики для анализа цветов
    color_analysis_data = []

    frame_idx = 0
    processed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Пропускать кадры для ускорения
        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue

        # Уменьшить размер кадра для обработки
        if resize_factor < 1.0:
            process_frame = cv2.resize(frame, (process_w, process_h))
        else:
            process_frame = frame

        # Распознавание времени
        timestamp_str = "00:00:00"
        current_dt = datetime.now()
        
        if not disable_ocr and reader is not None:
            try:
                y1 = max(0, min(ocr_y, frame.shape[0] - ocr_height))
                y2 = min(frame.shape[0], y1 + ocr_height)
                x1 = max(0, min(ocr_x, frame.shape[1] - ocr_width))
                x2 = min(frame.shape[1], x1 + ocr_width)
                
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
                blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                
                result = reader.readtext(blurred, detail=0, paragraph=True)
                
                if result:
                    timestamp_str = result[0]
                    time_formats = ["%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"]
                    
                    for fmt in time_formats:
                        try:
                            current_time = datetime.strptime(timestamp_str, fmt).time()
                            current_dt = datetime.combine(datetime.today(), current_time)
                            break
                        except:
                            continue
                            
            except Exception as e:
                st.sidebar.warning(f"Ошибка OCR: {e}")

        # Детекция объектов
        results = model.track(process_frame, persist=True, conf=confidence, 
                            classes=[0, 2, 3, 5, 6, 7], tracker="bytetrack.yaml", verbose=False)[0]
        
        # Создаем Detections объект
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidence_scores = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            tracker_ids = results.boxes.id.cpu().numpy().astype(int)
            
            # Масштабируем координаты обратно к исходному размеру
            if resize_factor < 1.0:
                boxes = boxes / resize_factor
            
            detections = Detections(
                xyxy=boxes,
                confidence=confidence_scores,
                class_id=class_ids,
                tracker_id=tracker_ids
            )
            
            tracks = tracker_ids
        else:
            detections = Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.array([]),
                class_id=np.array([], dtype=int)
            )
            tracks = None

        # УПРОЩЕННЫЙ ReID логика для людей
        current_people_tracks = []
        reid_matches = {}
        
        if enable_reid and reid_storage is not None and tracks is not None:
            # Фильтруем только людей
            people_indices = np.where(detections.class_id == 0)[0]
            
            for idx in people_indices:
                tracker_id = int(tracks[idx])
                bbox = detections.xyxy[idx]
                
                # Извлекаем особенности
                descriptors, histogram = reid_storage.extract_features(frame, bbox)
                
                if descriptors is not None or histogram is not None:
                    # Ищем совпадение в известных людях
                    matched_id, similarity = reid_storage.find_best_match(descriptors, histogram)
                    
                    if matched_id is not None and similarity > reid_threshold:
                        # Нашли совпадение - используем старый ID
                        reid_matches[tracker_id] = matched_id
                        reid_storage.update_person(matched_id, descriptors, histogram)
                        current_people_tracks.append(matched_id)
                    else:
                        # Новый человек или нет хорошего совпадения
                        if tracker_id not in id_mapping:
                            # Создаем новый постоянный ID
                            new_person_id = next_person_id
                            next_person_id += 1
                            id_mapping[tracker_id] = new_person_id
                            reid_storage.add_person(new_person_id, descriptors, histogram)
                            current_people_tracks.append(new_person_id)
                        else:
                            # Уже есть маппинг
                            current_people_tracks.append(id_mapping[tracker_id])
                else:
                    # Не удалось извлечь особенности, используем tracker_id
                    if tracker_id not in id_mapping:
                        new_person_id = next_person_id
                        next_person_id += 1
                        id_mapping[tracker_id] = new_person_id
                    current_people_tracks.append(id_mapping[tracker_id])
        
        # Текущее кол-во людей
        people_count = np.sum(detections.class_id == 0) if len(detections) > 0 else 0
        occupancy.append({"time": current_dt.strftime("%H:%M:%S"), "people": people_count})

        # Обновление данных людей с учетом ReID
        if tracks is not None:
            # Используем ReID ID если доступны, иначе tracker_id
            if enable_reid and current_people_tracks:
                current_ids = set(current_people_tracks)
            else:
                current_ids = set(tracks[detections.class_id == 0])
            
            appeared = current_ids - tracked_ids
            disappeared = tracked_ids - current_ids

            for person_id in appeared:
                people_data.append({
                    "ID": int(person_id), 
                    "Появление": current_dt.strftime("%H:%M:%S"), 
                    "Исчезновение": "-", 
                    "Ожидание": "-",
                    "ReID": "✓" if enable_reid else "✗"
                })
            
            for person_id in disappeared:
                for row in people_data:
                    if row["ID"] == int(person_id) and row["Исчезновение"] == "-":
                        row["Исчезновение"] = current_dt.strftime("%H:%M:%S")
                        try:
                            t1 = datetime.strptime(row["Появление"], "%H:%M:%S")
                            t2 = datetime.strptime(row["Исчезновение"], "%H:%M:%S")
                            wait = (t2 - t1).total_seconds() / 60
                            row["Ожидание"] = f"{wait:.1f} мин"
                        except:
                            row["Ожидание"] = "0.0 мин"

            tracked_ids = current_ids

        # Детекция поездов ТОЛЬКО в указанной зоне
        train_detected = False
        best_train_confidence = 0
        best_train_info = ""
        
        if len(detections) > 0:
            potential_train_indices = np.where((detections.class_id == 6) |  # train
                                             ((detections.class_id == 5) & (detections.confidence > 0.6)) |  # bus
                                             ((detections.class_id == 7) & (detections.confidence > 0.6)))[0]  # truck
            
            for idx in potential_train_indices:
                x1, y1, x2, y2 = detections.xyxy[idx].astype(int)
                width = x2 - x1
                height = y2 - y1
                
                in_train_zone, zone_coords = is_in_train_zone(x1, y1, x2, y2, original_w, original_h)
                
                if not in_train_zone:
                    continue
                
                if width > original_w * 0.25 and height > original_h * 0.15:
                    x1_clip = max(0, x1)
                    y1_clip = max(0, y1)
                    x2_clip = min(original_w, x2)
                    y2_clip = min(original_h, y2)
                    
                    roi = frame[y1_clip:y2_clip, x1_clip:x2_clip]
                    
                    if roi.size > 0:
                        gray_percent, orange_percent, red_percent, combined_percent = detect_train_colors(roi)
                        
                        color_analysis_data.append({
                            "frame": frame_idx,
                            "gray": gray_percent,
                            "orange": orange_percent,
                            "red": red_percent,
                            "combined": combined_percent
                        })
                        
                        is_train_by_color = (
                            (gray_percent > 10) or
                            (orange_percent > 5) or
                            (red_percent > 5) or
                            (combined_percent > 15)
                        )
                        
                        detection_confidence = detections.confidence[idx]
                        combined_confidence = detection_confidence * 0.7 + (combined_percent / 100) * 0.3
                        
                        if is_train_by_color and combined_confidence > 0.5:
                            if combined_confidence > best_train_confidence:
                                best_train_confidence = combined_confidence
                                best_train_info = f"G:{gray_percent:.1f}% O:{orange_percent:.1f}% R:{red_percent:.1f}%"
                            train_detected = True

        if train_detected and not train_present:
            train_arrival_time = current_dt.strftime("%H:%M:%S")
            train_present = True
            st.sidebar.success(f"🚂 Поезд обнаружен в {train_arrival_time}")
            st.sidebar.info(f"Цвета: {best_train_info}")
        elif not train_detected and train_present:
            train_events.append({
                "Прибытие": train_arrival_time, 
                "Убытие": current_dt.strftime("%H:%M:%S")
            })
            train_present = False
            st.sidebar.info(f"Поезд уехал в {current_dt.strftime('%H:%M:%S')}")

        # Линия входа/выхода
        line.trigger(detections=detections)
        
        # Визуализация зоны поезда на кадре
        zone_x1, zone_y1, zone_x2, zone_y2 = is_in_train_zone(0, 0, 0, 0, original_w, original_h)[1]
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 2)
        cv2.putText(frame, "TRAIN ZONE", (zone_x1, zone_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Аннотации
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = line_annotator.annotate(frame, line)

        # Отображение ID с учетом ReID
        if detections.tracker_id is not None:
            for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
                class_id = detections.class_id[i]
                tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else 0
                
                # Определяем ID для отображения
                if class_id == 0 and enable_reid:  # Человек с включенным ReID
                    display_id = id_mapping.get(tracker_id, tracker_id)
                    id_text = f"Person {display_id}"
                    color = (0, 255, 0)  # Зеленый для людей с ReID
                elif class_id == 0:  # Человек без ReID
                    display_id = tracker_id
                    id_text = f"Person {display_id}"
                    color = (255, 255, 255)  # Белый для людей без ReID
                elif class_id == 6:  # Поезд
                    class_names = {0: "Person", 2: "Car", 3: "Motorcycle", 5: "Bus", 6: "Train", 7: "Truck"}
                    class_name = class_names.get(class_id, "Unknown")
                    id_text = f"{class_name} {tracker_id}"
                    color = (0, 255, 255)  # Желтый для поездов
                else:
                    class_names = {0: "Person", 2: "Car", 3: "Motorcycle", 5: "Bus", 6: "Train", 7: "Truck"}
                    class_name = class_names.get(class_id, "Unknown")
                    id_text = f"{class_name} {tracker_id}"
                    color = (255, 255, 255)  # Белый для других объектов
                
                cv2.putText(frame, id_text, 
                           (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)

        # Надпись времени и информации
        cv2.putText(frame, f"Time: {timestamp_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"People: {people_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Train: {'Yes' if train_present else 'No'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"ReID: {'ON' if enable_reid else 'OFF'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if train_detected:
            cv2.putText(frame, f"Train Colors: {best_train_info}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        stframe.image(frame, channels="BGR")
        progress.progress(min(frame_idx / frame_count, 1.0))
        frame_idx += 1
        processed_frames += 1

    cap.release()

    # === Дашборд ===
    st.success(f"Обработка завершена! Обработано {processed_frames} кадров из {frame_count}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Люди на платформе")
        if people_data:
            df_people = pd.DataFrame(people_data)
            st.dataframe(df_people)
            csv_people = df_people.to_csv(index=False).encode()
            st.download_button("Скачать CSV людей", csv_people, "people.csv")
            
            # Статистика ReID
            if enable_reid and reid_storage:
                st.subheader("📊 Статистика ReID")
                reid_count = len(reid_storage.known_descriptors) if reid_storage else 0
                st.metric("Уникальных людей (ReID)", reid_count)
        else:
            st.write("Нет данных о людях")

    with col2:
        st.subheader("🚂 Поезда")
        if train_events:
            df_train = pd.DataFrame(train_events)
            st.dataframe(df_train)
            csv_train = df_train.to_csv(index=False).encode()
            st.download_button("Скачать CSV поездов", csv_train, "trains.csv")
        else:
            st.write("Нет данных о поездах")

    # Визуализация анализа цветов
    if color_analysis_data:
        st.subheader("🎨 Анализ цветов поезда")
        df_colors = pd.DataFrame(color_analysis_data)
        st.line_chart(df_colors[['gray', 'orange', 'red', 'combined']])

    st.subheader("📈 Загруженность платформы по времени")
    if occupancy:
        df_occ = pd.DataFrame(occupancy)
        st.line_chart(df_occ.set_index("time"))
    else:
        st.write("Нет данных о загруженности")

    if people_data:
        try:
            wait_times = [float(row["Ожидание"].replace(" мин", "")) for row in people_data if row["Ожидание"] != "-"]
            if wait_times:
                avg_wait = sum(wait_times) / len(wait_times)
                st.metric("Среднее время ожидания поезда", f"{avg_wait:.1f} мин")
        except:
            st.metric("Среднее время ожидания поезда", "Н/Д")

    st.subheader("Вход/Выход")
    st.write(f"Вошло: {line.in_count} Вышло: {line.out_count}")
    ///фыва