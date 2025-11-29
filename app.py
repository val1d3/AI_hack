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
import sqlite3
import json
from contextlib import contextmanager
import re

# --- Настройки ---
st.title("🚉 Анализ видео с железнодорожной платформы")
st.sidebar.header("Настройки")

# Настройки базы данных
db_enabled = st.sidebar.checkbox("Сохранять в базу данных", value=True)
db_path = st.sidebar.text_input("Путь к базе данных", "platform_analysis.db")

# Основные настройки
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

# ✅ УЛУЧШЕННЫЙ ReID с анализом деталей одежды
st.sidebar.subheader("Настройки ReID с анализом деталей")
reid_threshold = st.sidebar.slider("Порог ReID сходства", 0.1, 1.0, 0.7)
enable_reid = st.sidebar.checkbox("Включить ReID", value=True)
analyze_clothing = st.sidebar.checkbox("Анализировать одежду и обувь", value=True)
clothing_detail_level = st.sidebar.slider("Уровень детализации одежды", 1, 3, 2)

# Настройки анализа частей тела
st.sidebar.subheader("Анализ частей тела")
analyze_upper_body = st.sidebar.checkbox("Анализировать верхнюю часть", value=True)
analyze_lower_body = st.sidebar.checkbox("Анализировать нижнюю часть", value=True)
analyze_shoes = st.sidebar.checkbox("Анализировать обувь", value=True)
analyze_head = st.sidebar.checkbox("Анализировать голову", value=False)

# Дополнительные настройки
st.sidebar.subheader("Дополнительные настройки")
train_number_ocr_enabled = st.sidebar.checkbox("Распознавать номер поезда", value=True)

# --- Функции для расчета статистики времени ожидания ---
def calculate_total_waiting_time(people_data):
    """Суммирует общее время ожидания всех людей"""
    total_minutes = 0.0
    
    for person in people_data:
        waiting_str = person.get("Ожидание", "0.0 мин")
        
        # Извлекаем числовое значение из строки "X.X мин"
        try:
            # Убираем " мин" и преобразуем в float
            if "мин" in waiting_str:
                minutes = float(waiting_str.replace(" мин", "").strip())
                total_minutes += minutes
        except (ValueError, AttributeError):
            continue
    
    return total_minutes

def analyze_waiting_distribution(people_data):
    """Анализирует распределение времени ожидания"""
    distribution = {
        "менее_1_мин": 0,
        "1_5_мин": 0,
        "5_15_мин": 0,
        "более_15_мин": 0
    }
    
    for person in people_data:
        waiting_str = person.get("Ожидание", "0.0 мин")
        
        try:
            if "мин" in waiting_str:
                minutes = float(waiting_str.replace(" мин", "").strip())
                
                if minutes < 1:
                    distribution["менее_1_мин"] += 1
                elif 1 <= minutes < 5:
                    distribution["1_5_мин"] += 1
                elif 5 <= minutes < 15:
                    distribution["5_15_мин"] += 1
                else:
                    distribution["более_15_мин"] += 1
                    
        except (ValueError, AttributeError):
            continue
    
    return distribution

# --- База данных ---
@contextmanager
def get_db_connection():
    """Контекстный менеджер для подключения к БД"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Инициализация базы данных с проверкой существующих таблиц"""
    with get_db_connection() as conn:
        # Таблица для информации о видео
        conn.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                frame_count INTEGER,
                processed_frames INTEGER,
                duration_seconds REAL,
                resolution TEXT,
                fps REAL
            )
        ''')
        
        # Таблица для данных о людях
        conn.execute('''
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                person_id INTEGER NOT NULL,
                appearance_time TEXT NOT NULL,
                disappearance_time TEXT,
                waiting_minutes REAL,
                reid_enabled BOOLEAN,
                clothing_features TEXT,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Таблица для событий поездов
        conn.execute('''
            CREATE TABLE IF NOT EXISTS train_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                arrival_time TEXT NOT NULL,
                departure_time TEXT,
                duration_seconds REAL,
                train_number TEXT,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Таблица для загруженности платформы
        conn.execute('''
            CREATE TABLE IF NOT EXISTS occupancy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                timestamp TEXT NOT NULL,
                people_count INTEGER NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Таблица для анализа цветов поезда
        conn.execute('''
            CREATE TABLE IF NOT EXISTS color_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                frame_index INTEGER NOT NULL,
                gray_percent REAL,
                orange_percent REAL,
                red_percent REAL,
                combined_percent REAL,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Таблица для статистики входа/выхода
        conn.execute('''
            CREATE TABLE IF NOT EXISTS line_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                in_count INTEGER DEFAULT 0,
                out_count INTEGER DEFAULT 0,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Таблица для статистики времени пребывания
        conn.execute('''
            CREATE TABLE IF NOT EXISTS stay_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                total_stay_minutes REAL,
                average_stay_minutes REAL,
                median_stay_minutes REAL,
                max_stay_minutes REAL,
                min_stay_minutes REAL,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Таблица для статистики времени ожидания
        conn.execute('''
            CREATE TABLE IF NOT EXISTS waiting_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                total_waiting_minutes REAL,
                average_waiting_minutes REAL,
                less_1_min INTEGER,
                between_1_5_min INTEGER,
                between_5_15_min INTEGER,
                more_15_min INTEGER,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Таблица для статистики людей возле поезда
        conn.execute('''
            CREATE TABLE IF NOT EXISTS train_proximity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                timestamp TEXT NOT NULL,
                people_near_train INTEGER NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Таблица для настроек обработки
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processing_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                settings_json TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        ''')
        
        # Проверяем и добавляем отсутствующие столбцы
        try:
            # Проверяем наличие столбца clothing_features в таблице people
            cursor = conn.execute("PRAGMA table_info(people)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'clothing_features' not in columns:
                conn.execute('ALTER TABLE people ADD COLUMN clothing_features TEXT')
                st.sidebar.info("Добавлен отсутствующий столбец clothing_features в таблицу people")
                
            # Проверяем наличие столбца train_number в таблице train_events
            cursor = conn.execute("PRAGMA table_info(train_events)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'train_number' not in columns:
                conn.execute('ALTER TABLE train_events ADD COLUMN train_number TEXT')
                st.sidebar.info("Добавлен отсутствующий столбец train_number в таблицу train_events")
                
        except Exception as e:
            st.sidebar.warning(f"Ошибка при проверке структуры БД: {e}")
        
        conn.commit()

def save_video_info(conn, video_info):
    """Сохраняет информацию о видео"""
    cursor = conn.execute('''
        INSERT INTO videos (filename, frame_count, processed_frames, duration_seconds, resolution, fps)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        video_info['filename'],
        video_info['frame_count'],
        video_info['processed_frames'],
        video_info['duration_seconds'],
        video_info['resolution'],
        video_info['fps']
    ))
    return cursor.lastrowid

def save_people_data(conn, video_id, people_data):
    """Сохраняет данные о людях"""
    for person in people_data:
        waiting_minutes = None
        if person["Ожидание"] != "0.0 мин":
            try:
                waiting_minutes = float(person["Ожидание"].replace(" мин", ""))
            except:
                waiting_minutes = 0.0
        else:
            waiting_minutes = 0.0
        
        clothing_features = person.get("ClothingFeatures", "{}")
        
        # Проверяем наличие столбца clothing_features перед вставкой
        cursor = conn.execute("PRAGMA table_info(people)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'clothing_features' in columns:
            conn.execute('''
                INSERT INTO people (video_id, person_id, appearance_time, disappearance_time, waiting_minutes, reid_enabled, clothing_features)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_id,
                person["ID"],
                person["Появление"],
                person["Исчезновение"] if person["Исчезновение"] != "-" else None,
                waiting_minutes,
                person["ReID"] == "✓",
                clothing_features
            ))
        else:
            # Если столбца нет, вставляем без него
            conn.execute('''
                INSERT INTO people (video_id, person_id, appearance_time, disappearance_time, waiting_minutes, reid_enabled)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                video_id,
                person["ID"],
                person["Появление"],
                person["Исчезновение"] if person["Исчезновение"] != "-" else None,
                waiting_minutes,
                person["ReID"] == "✓"
            ))

def save_train_events(conn, video_id, train_events):
    """Сохраняет события поездов"""
    for event in train_events:
        duration = None
        if event["Прибытие"] and event["Убытие"]:
            try:
                t1 = datetime.strptime(event["Прибытие"], "%H:%M:%S")
                t2 = datetime.strptime(event["Убытие"], "%H:%M:%S")
                duration = (t2 - t1).total_seconds()
            except:
                duration = None
        
        conn.execute('''
            INSERT INTO train_events (video_id, arrival_time, departure_time, duration_seconds, train_number)
            VALUES (?, ?, ?, ?, ?)
        ''', (video_id, event["Прибытие"], event["Убытие"], duration, event.get("Номер", None)))

def save_occupancy(conn, video_id, occupancy):
    """Сохраняет данные о загруженности"""
    for occ in occupancy:
        conn.execute('''
            INSERT INTO occupancy (video_id, timestamp, people_count)
            VALUES (?, ?, ?)
        ''', (video_id, occ["time"], occ["people"]))

def save_color_analysis(conn, video_id, color_analysis):
    """Сохраняет анализ цветов"""
    for color_data in color_analysis:
        conn.execute('''
            INSERT INTO color_analysis (video_id, frame_index, gray_percent, orange_percent, red_percent, combined_percent)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            video_id,
            color_data["frame"],
            color_data["gray"],
            color_data["orange"],
            color_data["red"],
            color_data["combined"]
        ))

def save_line_statistics(conn, video_id, in_count, out_count):
    """Сохраняет статистику линии"""
    conn.execute('''
        INSERT INTO line_statistics (video_id, in_count, out_count)
        VALUES (?, ?, ?)
    ''', (video_id, in_count, out_count))

def save_stay_statistics(conn, video_id, stay_stats):
    """Сохраняет статистику времени пребывания"""
    conn.execute('''
        INSERT INTO stay_statistics (video_id, total_stay_minutes, average_stay_minutes, median_stay_minutes, max_stay_minutes, min_stay_minutes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        video_id,
        stay_stats['total_stay_minutes'],
        stay_stats['average_stay_minutes'],
        stay_stats['median_stay_minutes'],
        stay_stats['max_stay_minutes'],
        stay_stats['min_stay_minutes']
    ))

def save_waiting_statistics(conn, video_id, waiting_stats):
    """Сохраняет статистику времени ожидания"""
    conn.execute('''
        INSERT INTO waiting_statistics (video_id, total_waiting_minutes, average_waiting_minutes, less_1_min, between_1_5_min, between_5_15_min, more_15_min)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        video_id,
        waiting_stats['total_waiting_minutes'],
        waiting_stats['average_waiting_minutes'],
        waiting_stats['less_1_min'],
        waiting_stats['between_1_5_min'],
        waiting_stats['between_5_15_min'],
        waiting_stats['more_15_min']
    ))

def save_train_proximity(conn, video_id, proximity_data):
    """Сохраняет данные о людях возле поезда"""
    for data in proximity_data:
        conn.execute('''
            INSERT INTO train_proximity (video_id, timestamp, people_near_train)
            VALUES (?, ?, ?)
        ''', (video_id, data["time"], data["people_near_train"]))

def save_processing_settings(conn, video_id, settings):
    """Сохраняет настройки обработки"""
    conn.execute('''
        INSERT INTO processing_settings (video_id, settings_json)
        VALUES (?, ?)
    ''', (video_id, json.dumps(settings)))

def load_previous_analyses():
    """Загружает список предыдущих анализов"""
    if not os.path.exists(db_path):
        return []
    
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT id, filename, processed_date, frame_count, processed_frames 
                FROM videos 
                ORDER BY processed_date DESC
            ''')
            return cursor.fetchall()
    except:
        return []

# Инициализация базы данных
if db_enabled:
    init_database()

# Модель
model = YOLO("yolov8n.pt")

# OCR ридер
reader = None
if not disable_ocr:
    try:
        reader = easyocr.Reader(['en', 'ru'], gpu=False)
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
train_proximity_data = []

# ✅ УЛУЧШЕННЫЙ ReID с анализом деталей одежды
class AdvancedReIDStorage:
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.known_features = OrderedDict()  # ID -> словарь с признаками
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def extract_detailed_features(self, image, bbox):
        """Извлекает детальные особенности одежды и обуви"""
        try:
            # Вырезаем область человека
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            person_crop = image[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return None
            
            height, width = person_crop.shape[:2]
            
            # ✅ РАЗДЕЛЕНИЕ НА ЧАСТИ ТЕЛА ДЛЯ АНАЛИЗА ДЕТАЛЕЙ
            parts_features = {}
            
            # 1. Верхняя часть тела (голова и торс)
            if analyze_upper_body and height > 40:
                upper_height = int(height * 0.6)  # Верхние 60%
                upper_part = person_crop[0:upper_height, :]
                parts_features['upper_body'] = self.analyze_clothing_region(upper_part, "upper")
            
            # 2. Нижняя часть тела (брюки/юбка)
            if analyze_lower_body and height > 40:
                lower_start = int(height * 0.4)  # Нижние 60%
                lower_part = person_crop[lower_start:, :]
                parts_features['lower_body'] = self.analyze_clothing_region(lower_part, "lower")
            
            # 3. Обувь (самые нижние 20%)
            if analyze_shoes and height > 50:
                shoes_start = int(height * 0.8)  # Нижние 20%
                shoes_part = person_crop[shoes_start:, :]
                parts_features['shoes'] = self.analyze_shoes_region(shoes_part)
            
            # 4. Голова (верхние 25%)
            if analyze_head and height > 40:
                head_height = int(height * 0.25)
                head_part = person_crop[0:head_height, :]
                parts_features['head'] = self.analyze_head_region(head_part)
            
            # 5. Общие особенности всего тела
            parts_features['whole_body'] = self.analyze_whole_body(person_crop)
            
            return parts_features
        except Exception as e:
            return None
    
    def analyze_clothing_region(self, region, region_type):
        """Анализирует регион одежды"""
        features = {}
        
        # Цветовые гистограммы в разных пространствах
        features['color_hsv'] = self.compute_color_histogram(region, 'HSV')
        features['color_lab'] = self.compute_color_histogram(region, 'LAB')
        features['color_rgb'] = self.compute_color_histogram(region, 'RGB')
        
        # Текстура с помощью LBP (Local Binary Patterns)
        features['texture'] = self.compute_texture_features(region)
        
        # ORB особенности для паттернов одежды
        features['orb_descriptors'] = self.compute_orb_features(region)
        
        # Доминирующие цвета
        features['dominant_colors'] = self.extract_dominant_colors(region)
        
        return features
    
    def analyze_shoes_region(self, region):
        """Специализированный анализ обуви"""
        features = {}
        
        # Обувь часто имеет характерные цвета
        features['color_hsv'] = self.compute_color_histogram(region, 'HSV')
        features['color_lab'] = self.compute_color_histogram(region, 'LAB')
        
        # Текстура для разных типов обуви (кожа, ткань и т.д.)
        features['texture'] = self.compute_texture_features(region)
        
        # Особенности формы (если обувь видна четко)
        features['shape_features'] = self.compute_shape_features(region)
        
        return features
    
    def analyze_head_region(self, region):
        """Анализ головы (волосы, головные уборы)"""
        features = {}
        
        features['color_hsv'] = self.compute_color_histogram(region, 'HSV')
        features['color_lab'] = self.compute_color_histogram(region, 'LAB')
        features['texture'] = self.compute_texture_features(region)
        
        return features
    
    def analyze_whole_body(self, region):
        """Анализ всего тела"""
        features = {}
        
        features['color_hsv'] = self.compute_color_histogram(region, 'HSV')
        features['color_lab'] = self.compute_color_histogram(region, 'LAB')
        features['orb_descriptors'] = self.compute_orb_features(region)
        features['texture'] = self.compute_texture_features(region)
        
        # Соотношение сторон (может помочь в идентификации телосложения)
        features['aspect_ratio'] = region.shape[1] / region.shape[0] if region.shape[0] > 0 else 0
        
        return features
    
    def compute_color_histogram(self, image, color_space='HSV'):
        """Вычисляет цветовую гистограмму в заданном пространстве"""
        try:
            if color_space == 'HSV':
                converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                channels = [0, 1, 2]
                hist_size = [8, 8, 8]
                ranges = [0, 180, 0, 256, 0, 256]
            elif color_space == 'LAB':
                converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                channels = [0, 1, 2]
                hist_size = [8, 8, 8]
                ranges = [0, 256, 0, 256, 0, 256]
            else:  # RGB
                converted = image
                channels = [0, 1, 2]
                hist_size = [8, 8, 8]
                ranges = [0, 256, 0, 256, 0, 256]
            
            hist = cv2.calcHist([converted], channels, None, hist_size, ranges)
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except:
            return np.array([])
    
    def compute_texture_features(self, image):
        """Вычисляет особенности текстуры с помощью LBP"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # LBP (Local Binary Patterns)
            radius = 3
            n_points = 8 * radius
            lbp = self.local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Гистограмма LBP
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)  # Нормализация
            
            return hist
        except:
            return np.array([])
    
    def local_binary_pattern(self, image, num_points, radius, method='uniform'):
        """Реализация Local Binary Pattern"""
        # Упрощенная версия LBP
        gray = image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        lbp = np.zeros_like(gray)
        for i in range(radius, gray.shape[0]-radius):
            for j in range(radius, gray.shape[1]-radius):
                center = gray[i,j]
                binary_code = 0
                for k in range(num_points):
                    angle = 2 * np.pi * k / num_points
                    x = i + int(radius * np.sin(angle))
                    y = j + int(radius * np.cos(angle))
                    if gray[x,y] >= center:
                        binary_code |= (1 << k)
                lbp[i,j] = binary_code
        
        return lbp
    
    def compute_orb_features(self, image):
        """Извлекает ORB дескрипторы"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            return descriptors
        except:
            return None
    
    def compute_shape_features(self, region):
        """Вычисляет особенности формы (для обуви)"""
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Моменты формы
            moments = cv2.moments(edges)
            
            features = []
            if moments['m00'] != 0:
                # Центр масс
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                features.extend([cx, cy])
            
            # Hu moments (инварианты к масштабу и вращению)
            hu_moments = cv2.HuMoments(moments)
            if hu_moments is not None:
                features.extend(hu_moments.flatten())
            
            return np.array(features)
        except:
            return np.array([])
    
    def extract_dominant_colors(self, image, k=3):
        """Извлекает доминирующие цвета с помощью k-means"""
        try:
            pixels = image.reshape(-1, 3)
            pixels = np.float32(pixels)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Преобразуем обратно в uint8
            centers = np.uint8(centers)
            
            # Сортируем по распространенности
            unique, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(-counts)
            dominant_colors = centers[sorted_indices]
            
            return dominant_colors.flatten()
        except:
            return np.array([])
    
    def calculate_features_similarity(self, features1, features2):
        """Вычисляет схожесть между двумя наборами признаков"""
        if features1 is None or features2 is None:
            return 0.0
        
        total_similarity = 0.0
        weight_sum = 0.0
        
        # Сравниваем каждую часть тела
        for part in ['upper_body', 'lower_body', 'shoes', 'head', 'whole_body']:
            if part in features1 and part in features2:
                part_similarity = self.compare_part_features(features1[part], features2[part])
                
                # Веса для разных частей тела
                weights = {
                    'upper_body': 0.3,
                    'lower_body': 0.25,
                    'shoes': 0.2,
                    'head': 0.15,
                    'whole_body': 0.1
                }
                
                total_similarity += part_similarity * weights.get(part, 0.1)
                weight_sum += weights.get(part, 0.1)
        
        if weight_sum > 0:
            return total_similarity / weight_sum
        else:
            return 0.0
    
    def compare_part_features(self, part1, part2):
        """Сравнивает особенности конкретной части тела"""
        similarity = 0.0
        feature_count = 0
        
        # Сравниваем цветовые гистограммы
        for color_space in ['color_hsv', 'color_lab', 'color_rgb']:
            if color_space in part1 and color_space in part2:
                if len(part1[color_space]) > 0 and len(part2[color_space]) > 0:
                    try:
                        color_sim = cv2.compareHist(part1[color_space].astype(np.float32), 
                                                   part2[color_space].astype(np.float32), 
                                                   cv2.HISTCMP_CORREL)
                        if not np.isnan(color_sim):
                            similarity += max(0, color_sim) * 0.3
                            feature_count += 0.3
                    except:
                        pass
        
        # Сравниваем текстуру
        if 'texture' in part1 and 'texture' in part2:
            if len(part1['texture']) > 0 and len(part2['texture']) > 0:
                try:
                    texture_sim = 1 - np.linalg.norm(part1['texture'] - part2['texture'])
                    similarity += max(0, texture_sim) * 0.3
                    feature_count += 0.3
                except:
                    pass
        
        # Сравниваем ORB дескрипторы
        if 'orb_descriptors' in part1 and 'orb_descriptors' in part2:
            desc1 = part1['orb_descriptors']
            desc2 = part2['orb_descriptors']
            if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
                try:
                    matches = self.bf.match(desc1, desc2)
                    if len(matches) > 0:
                        orb_sim = len(matches) / min(len(desc1), len(desc2))
                        similarity += min(orb_sim, 1.0) * 0.2
                        feature_count += 0.2
                except:
                    pass
        
        # Сравниваем доминирующие цвета
        if 'dominant_colors' in part1 and 'dominant_colors' in part2:
            if len(part1['dominant_colors']) > 0 and len(part2['dominant_colors']) > 0:
                try:
                    color_dist = np.linalg.norm(part1['dominant_colors'] - part2['dominant_colors'])
                    color_sim = 1 - min(color_dist / 100, 1.0)  # Нормализуем расстояние
                    similarity += max(0, color_sim) * 0.2
                    feature_count += 0.2
                except:
                    pass
        
        if feature_count > 0:
            return similarity / feature_count
        else:
            return 0.0
    
    def find_best_match(self, new_features):
        """Находит лучшего кандидата для повторной идентификации"""
        if not self.known_features:
            return None, 0.0
            
        best_match_id = None
        best_similarity = 0.0
        
        for person_id, stored_features in self.known_features.items():
            similarity = self.calculate_features_similarity(new_features, stored_features)
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = person_id
                
        return best_match_id, best_similarity
    
    def add_person(self, person_id, features):
        """Добавляет нового человека в хранилище"""
        self.known_features[person_id] = features
    
    def update_person(self, person_id, new_features):
        """Обновляет данные существующего человека"""
        if person_id in self.known_features:
            # Скользящее среднее для обновления признаков
            old_features = self.known_features[person_id]
            alpha = 0.7  # Коэффициент забывания
            
            updated_features = {}
            for part in ['upper_body', 'lower_body', 'shoes', 'head', 'whole_body']:
                if part in old_features and part in new_features:
                    updated_features[part] = {}
                    
                    # Обновляем числовые признаки
                    for feature_type in old_features[part]:
                        if feature_type in new_features[part]:
                            old_val = old_features[part][feature_type]
                            new_val = new_features[part][feature_type]
                            
                            if isinstance(old_val, np.ndarray) and isinstance(new_val, np.ndarray):
                                if len(old_val) == len(new_val):
                                    updated_features[part][feature_type] = (
                                        alpha * old_val + (1 - alpha) * new_val
                                    )
                            else:
                                # Для нечисловых признаков используем новые
                                updated_features[part][feature_type] = new_val
                elif part in new_features:
                    updated_features[part] = new_features[part]
            
            self.known_features[person_id] = updated_features
        else:
            self.add_person(person_id, new_features)

# Инициализация улучшенного ReID
reid_storage = AdvancedReIDStorage(similarity_threshold=reid_threshold) if enable_reid else None

# Функции для цветовой фильтрации поезда
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
    zone_x2 = train_zone_x + train_zone_width
    zone_y1 = train_zone_y
    zone_y2 = train_zone_y + train_zone_height

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    in_zone = (zone_x1 <= center_x <= zone_x2) and (zone_y1 <= center_y <= zone_y2)
    return in_zone, (zone_x1, zone_y1, zone_x2, zone_y2)

# ✅ ДОБАВЛЕНО: Функция для расчета статистики времени пребывания
def calculate_stay_statistics(people_data, occupancy):
    """Вычисляет статистику времени пребывания всех людей"""
    if not people_data:
        return {
            'total_stay_minutes': 0,
            'average_stay_minutes': 0,
            'median_stay_minutes': 0,
            'max_stay_minutes': 0,
            'min_stay_minutes': 0,
            'total_people': 0
        }
    
    # Получаем последнее время из occupancy для тех, кто не ушел
    last_time_str = occupancy[-1]["time"] if occupancy else "00:00:00"
    
    stay_times = []
    
    for person in people_data:
        if person["Исчезновение"] != "-":
            # У человека есть время ухода
            try:
                start_time = datetime.strptime(person["Появление"], "%H:%M:%S")
                end_time = datetime.strptime(person["Исчезновение"], "%H:%M:%S")
                
                # Если время ухода меньше времени прихода, предполагаем что это следующий день
                if end_time < start_time:
                    end_time = end_time.replace(day=end_time.day + 1)
                
                stay_seconds = (end_time - start_time).total_seconds()
                stay_minutes = stay_seconds / 60
                stay_times.append(stay_minutes)
                
            except Exception as e:
                continue
    
    if not stay_times:
        return {
            'total_stay_minutes': 0,
            'average_stay_minutes': 0,
            'median_stay_minutes': 0,
            'max_stay_minutes': 0,
            'min_stay_minutes': 0,
            'total_people': len(people_data)
        }
    
    # Вычисляем статистику
    total_stay_minutes = sum(stay_times)
    average_stay_minutes = total_stay_minutes / len(stay_times)
    median_stay_minutes = np.median(stay_times)
    max_stay_minutes = max(stay_times)
    min_stay_minutes = min(stay_times)
    
    return {
        'total_stay_minutes': total_stay_minutes,
        'average_stay_minutes': average_stay_minutes,
        'median_stay_minutes': median_stay_minutes,
        'max_stay_minutes': max_stay_minutes,
        'min_stay_minutes': min_stay_minutes,
        'total_people': len(people_data),
        'people_with_complete_data': len(stay_times)
    }

# --- Панель управления базами данных ---
st.sidebar.header("Управление данными")

if db_enabled:
    # Показать предыдущие анализы
    previous_analyses = load_previous_analyses()
    if previous_analyses:
        st.sidebar.subheader("Предыдущие анализы")
        for analysis in previous_analyses:
            st.sidebar.write(f"{analysis['filename']} - {analysis['processed_date']}")

# Загрузка видео
uploaded_file = st.file_uploader("Загрузи видео с платформы", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Сохраняем настройки для базы данных
    processing_settings = {
        "confidence": confidence,
        "line_y": line_y,
        "skip_frames": skip_frames,
        "resize_factor": resize_factor,
        "disable_ocr": disable_ocr,
        "ocr_x": ocr_x,
        "ocr_y": ocr_y,
        "ocr_width": ocr_width,
        "ocr_height": ocr_height,
        "reid_threshold": reid_threshold,
        "enable_reid": enable_reid,
        "analyze_clothing": analyze_clothing,
        "clothing_detail_level": clothing_detail_level,
        "analyze_upper_body": analyze_upper_body,
        "analyze_lower_body": analyze_lower_body,
        "analyze_shoes": analyze_shoes,
        "analyze_head": analyze_head,
        "train_number_ocr_enabled": train_number_ocr_enabled
    }
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # === АВТОМАТИЧЕСКАЯ ЗОНА ПОЕЗДА (ПРАВАЯ ЧАСТЬ КАДРА) ===
    # Поезд всегда справа и в нижних ~70% кадра
    train_zone_x = int(original_w * 0.55)          # начиная с 55% ширины
    train_zone_width = original_w - train_zone_x    # до конца кадра
    train_zone_y = int(original_h * 0.25)           # от 25% высоты (чтобы не цеплять небо/крышу)
    train_zone_height = original_h - train_zone_y
    
    st.sidebar.info(f"Автоматическая зона поезда: {train_zone_x}x{train_zone_y} - {train_zone_width}x{train_zone_height}")
    
    process_w = int(original_w * resize_factor)
    process_h = int(original_h * resize_factor)

    stframe = st.empty()
    progress = st.progress(0)

    # Для поезда
    train_present = False
    train_arrival_time = None
    train_number = None
    tracked_ids = set()
    
    # Для статистики людей возле поезда
    max_people_near_train = 0
    
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

        # ✅ УЛУЧШЕННЫЙ ReID с анализом деталей одежды
        current_people_tracks = []
        reid_matches = {}
        
        if enable_reid and reid_storage is not None and tracks is not None and analyze_clothing:
            # Фильтруем только людей
            people_indices = np.where(detections.class_id == 0)[0]
            
            for idx in people_indices:
                tracker_id = int(tracks[idx])
                bbox = detections.xyxy[idx]
                
                # Извлекаем детальные особенности одежды
                features = reid_storage.extract_detailed_features(frame, bbox)
                
                if features is not None:
                    # Ищем совпадение в известных людях
                    matched_id, similarity = reid_storage.find_best_match(features)
                    
                    if matched_id is not None and similarity > reid_threshold:
                        # Нашли совпадение - используем старый ID
                        reid_matches[tracker_id] = matched_id
                        reid_storage.update_person(matched_id, features)
                        current_people_tracks.append(matched_id)
                        
                        # Сохраняем информацию об одежде для отображения
                        clothing_info = f"Match: {similarity:.2f}"
                    else:
                        # Новый человек или нет хорошего совпадения
                        if tracker_id not in id_mapping:
                            # Создаем новый постоянный ID
                            new_person_id = next_person_id
                            next_person_id += 1
                            id_mapping[tracker_id] = new_person_id
                            reid_storage.add_person(new_person_id, features)
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
            if enable_reid and analyze_clothing and current_people_tracks:
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
                    "Ожидание": "0.0 мин",  # ИСПРАВЛЕНО: правильный формат
                    "ReID": "✓" if enable_reid and analyze_clothing else "✗",
                    "ClothingFeatures": json.dumps({"analyzed": analyze_clothing})
                })
            
            for person_id in disappeared:
                for row in people_data:
                    if row["ID"] == int(person_id) and row["Исчезновение"] == "-":
                        row["Исчезновение"] = current_dt.strftime("%H:%M:%S")
                        try:
                            t1 = datetime.strptime(row["Появление"], "%H:%M:%S")
                            t2 = datetime.strptime(row["Исчезновение"], "%H:%M:%S")
                            
                            # ИСПРАВЛЕНО: правильный расчет разницы времени
                            if t2 < t1:
                                # Если время ухода меньше времени прихода, добавляем 1 день
                                t2 = t2 + timedelta(days=1)
                            
                            wait_seconds = (t2 - t1).total_seconds()
                            wait_minutes = wait_seconds / 60
                            
                            # Форматируем правильно - исправляем "мм" на "мин"
                            row["Ожидание"] = f"{wait_minutes:.1f} мин"
                        except Exception as e:
                            row["Ожидание"] = "0.0 мин"

            tracked_ids = current_ids

        # === УЛУЧШЕННАЯ ДЕТЕКЦИЯ ПОЕЗДА ===
        train_detected = False
        best_train_confidence = 0
        best_train_info = ""

        if len(detections) > 0:
            # Ищем большие объекты справа
            for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
                cls_id = int(detections.class_id[i])
                conf = detections.confidence[i]
                area = (x2 - x1) * (y2 - y1)
                
                # Фильтр: только большие объекты
                if area < original_w * original_h * 0.08:  # минимум 8% кадра
                    continue
                    
                in_train_zone, _ = is_in_train_zone(x1, y1, x2, y2, original_w, original_h)
                if not in_train_zone:
                    continue

                # Приоритет: train > bus > truck > car (если очень большой)
                if cls_id == 6:  # train
                    score = conf * 1.3
                elif cls_id == 5 or cls_id == 7:  # bus / truck
                    score = conf * 1.1
                elif cls_id == 2 and area > original_w * original_h * 0.2:  # огромный car → вероятно поезд
                    score = conf * 0.9
                else:
                    continue

                # Дополнительно проверяем цвет
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                x1i, y1i = max(0, x1i), max(0, y1i)
                x2i, y2i = min(original_w, x2i), min(original_h, y2i)
                roi = frame[y1i:y2i, x1i:x2i]
                
                if roi.size > 0:
                    gray_p, orange_p, red_p, combined_p = detect_train_colors(roi)
                    color_bonus = combined_p / 100 * 0.4
                    final_score = score + color_bonus

                    if final_score > best_train_confidence:
                        best_train_confidence = final_score
                        best_train_info = f"G:{gray_p:.0f}% O:{orange_p:.0f}% R:{red_p:.0f}% C:{combined_p:.0f}%"
                        train_detected = True

        # Распознавание номера поезда
        current_train_number = None
        if train_detected and train_number_ocr_enabled and reader is not None:
            # Область для номера поезда — обычно вверху вагона или сбоку
            number_roi_x = int(original_w * 0.6)
            number_roi_y = int(original_h * 0.3)
            number_roi_w = int(original_w * 0.35)
            number_roi_h = int(original_h * 0.15)
            
            crop = frame[number_roi_y:number_roi_y+number_roi_h, 
                         number_roi_x:number_roi_x+number_roi_w]
            
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
                result = reader.readtext(enhanced, allowlist='0123456789АВЕКМНОРСТУХABEKMHOPCTYX', detail=0)
                
                if result:
                    text = " ".join(result).upper()
                    # Фильтруем типичные номера электропоездов
                    match = re.search(r'\b[А-ЯA-Z]{0,3}\d{3,4}[А-ЯA-Z]?\b', text)
                    if match:
                        current_train_number = match.group(0)
                    else:
                        current_train_number = result[0] if len(result) > 0 else None

            if current_train_number:
                train_number = current_train_number
                cv2.putText(frame, f"Поезд: {train_number}", (10, 210), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                st.sidebar.success(f"Номер поезда: {train_number}")

        # Обработка событий поезда
        if train_detected and not train_present:
            train_arrival_time = current_dt.strftime("%H:%M:%S")
            train_present = True
            st.sidebar.success(f"🚂 Поезд обнаружен в {train_arrival_time}")
            st.sidebar.info(f"Цвета: {best_train_info}")
            if train_number:
                st.sidebar.info(f"Номер поезда: {train_number}")
        elif not train_detected and train_present:
            train_events.append({
                "Прибытие": train_arrival_time, 
                "Убытие": current_dt.strftime("%H:%M:%S"),
                "Номер": train_number
            })
            train_present = False
            train_number = None
            st.sidebar.info(f"Поезд уехал в {current_dt.strftime('%H:%M:%S')}")

        # === ОПРЕДЕЛЕНИЕ ЛЮДЕЙ ВОЗЛЕ ПОЕЗДА ===
        people_near_train = 0
        near_train_ids = []

        if train_detected:
            train_left = train_zone_x
            buffer = int(original_w * 0.08)  # ~150px при FullHD
            
            for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
                if detections.class_id[i] != 0:  # не человек
                    continue
                person_center_x = (x1 + x2) / 2
                
                if person_center_x >= train_left - buffer:
                    people_near_train += 1
                    # Опционально: сохраняем ID
                    tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else 0
                    display_id = id_mapping.get(tracker_id, tracker_id) if enable_reid else tracker_id
                    near_train_ids.append(display_id)
                    
                    # Подсвечиваем таких людей красным
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, "NEAR TRAIN", (int(x1), int(y1)-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Обновляем максимум
            if people_near_train > max_people_near_train:
                max_people_near_train = people_near_train

        # Сохраняем данные о близости к поезду
        train_proximity_data.append({
            "time": current_dt.strftime("%H:%M:%S"), 
            "people_near_train": people_near_train
        })

        # Линия входа/выхода
        line.trigger(detections=detections)
        
        # Визуализация зоны поезда на кадре
        zone_x1, zone_y1, zone_x2, zone_y2 = is_in_train_zone(0, 0, 0, 0, original_w, original_h)[1]
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 2)
        cv2.putText(frame, "TRAIN ZONE", (zone_x1, zone_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Аннотации
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = line_annotator.annotate(frame, line)

        # Отображение ID с учетом ReID и информации об одежде
        if detections.tracker_id is not None:
            for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
                class_id = detections.class_id[i]
                tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else 0
                
                # Определяем ID для отображения
                if class_id == 0 and enable_reid and analyze_clothing:  # Человек с включенным ReID
                    display_id = id_mapping.get(tracker_id, tracker_id)
                    
                    # Получаем информацию о совпадении
                    match_info = ""
                    if tracker_id in reid_matches:
                        matched_id = reid_matches[tracker_id]
                        match_info = f" (Matched: {matched_id})"
                    
                    id_text = f"Person {display_id}{match_info}"
                    color = (0, 255, 0)  # Зеленый для людей с ReID
                    
                    # Рисуем bounding box с информацией об одежде
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
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
        cv2.putText(frame, f"Clothing Analysis: {'ON' if analyze_clothing else 'OFF'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if train_detected:
            cv2.putText(frame, f"Train Colors: {best_train_info}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if train_number:
            cv2.putText(frame, f"Поезд: {train_number}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Выводим на экран информацию о людях возле поезда
        cv2.putText(frame, f"Возле поезда: {people_near_train}", (10, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR")
        progress.progress(min(frame_idx / frame_count, 1.0))
        frame_idx += 1
        processed_frames += 1

    cap.release()

    # ✅ ДОБАВЛЕНО: Расчет статистики времени пребывания
    stay_statistics = calculate_stay_statistics(people_data, occupancy)
    
    # ✅ ДОБАВЛЕНО: Расчет статистики времени ожидания
    total_waiting_time = calculate_total_waiting_time(people_data)
    waiting_distribution = analyze_waiting_distribution(people_data)
    total_people = len(people_data)
    
    # Создаем статистику времени ожидания для базы данных
    waiting_stats = {
        'total_waiting_minutes': total_waiting_time,
        'average_waiting_minutes': total_waiting_time / total_people if total_people > 0 else 0,
        'less_1_min': waiting_distribution["менее_1_мин"],
        'between_1_5_min': waiting_distribution["1_5_мин"],
        'between_5_15_min': waiting_distribution["5_15_мин"],
        'more_15_min': waiting_distribution["более_15_мин"]
    }

    # === Сохранение в базу данных ===
    if db_enabled:
        try:
            with get_db_connection() as conn:
                # Сохраняем информацию о видео
                video_info = {
                    'filename': uploaded_file.name,
                    'frame_count': frame_count,
                    'processed_frames': processed_frames,
                    'duration_seconds': duration,
                    'resolution': f"{original_w}x{original_h}",
                    'fps': fps
                }
                
                video_id = save_video_info(conn, video_info)
                
                # Сохраняем все данные
                save_people_data(conn, video_id, people_data)
                save_train_events(conn, video_id, train_events)
                save_occupancy(conn, video_id, occupancy)
                save_color_analysis(conn, video_id, color_analysis_data)
                save_line_statistics(conn, video_id, line.in_count, line.out_count)
                save_stay_statistics(conn, video_id, stay_statistics)
                save_waiting_statistics(conn, video_id, waiting_stats)
                save_train_proximity(conn, video_id, train_proximity_data)
                save_processing_settings(conn, video_id, processing_settings)
                
                conn.commit()
                
                st.success(f"✅ Данные успешно сохранены в базу данных (ID: {video_id})")
                
        except Exception as e:
            st.error(f"❌ Ошибка при сохранении в базу данных: {e}")

    # === Дашборд ===
    st.success(f"Обработка завершена! Обработано {processed_frames} кадров из {frame_count}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Люди на платформе")
        if people_data:
            df_people = pd.DataFrame(people_data)
            # Убираем столбец с бинарными данными одежды для лучшего отображения
            display_columns = [col for col in df_people.columns if col != "ClothingFeatures"]
            st.dataframe(df_people[display_columns])
            csv_people = df_people.to_csv(index=False).encode()
            st.download_button("Скачать CSV людей", csv_people, "people.csv")
            
            # Статистика ReID
            if enable_reid and reid_storage and analyze_clothing:
                st.subheader("📊 Статистика ReID с анализом одежды")
                reid_count = len(reid_storage.known_features) if reid_storage else 0
                st.metric("Уникальных людей (ReID)", reid_count)
                st.metric("Уровень детализации", clothing_detail_level)

    with col2:
        st.subheader("🚂 Поезда")
        if train_events:
            df_train = pd.DataFrame(train_events)
            st.dataframe(df_train)
            csv_train = df_train.to_csv(index=False).encode()
            st.download_button("Скачать CSV поездов", csv_train, "trains.csv")
        else:
            st.write("Нет данных о поездах")

    # ✅ ДОБАВЛЕНО: Статистика времени пребывания
    st.subheader("⏱️ Статистика времени пребывания")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Общее время пребывания", f"{stay_statistics['total_stay_minutes']:.1f} мин")
        st.metric("Среднее время пребывания", f"{stay_statistics['average_stay_minutes']:.1f} мин")
    
    with col2:
        st.metric("Максимальное время", f"{stay_statistics['max_stay_minutes']:.1f} мин")
        st.metric("Минимальное время", f"{stay_statistics['min_stay_minutes']:.1f} мин")
    
    with col3:
        st.metric("Медианное время", f"{stay_statistics['median_stay_minutes']:.1f} мин")
        st.metric("Всего людей", stay_statistics['total_people'])

    # ✅ ДОБАВЛЕНО: Суммарная статистика времени ожидания
    st.subheader("📊 Суммарная статистика времени ожидания")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Общее время ожидания", f"{total_waiting_time:.1f} мин")
        st.metric("Всего людей", total_people)

    with col2:
        if total_people > 0:
            avg_waiting = total_waiting_time / total_people
            st.metric("Среднее время ожидания", f"{avg_waiting:.1f} мин")
        else:
            st.metric("Среднее время ожидания", "0.0 мин")

    with col3:
        # Находим максимальное время ожидания
        max_waiting = 0.0
        for person in people_data:
            waiting_str = person.get("Ожидание", "0.0 мин")
            try:
                if "мин" in waiting_str:
                    minutes = float(waiting_str.replace(" мин", "").strip())
                    if minutes > max_waiting:
                        max_waiting = minutes
            except:
                continue
        st.metric("Максимальное ожидание", f"{max_waiting:.1f} мин")

    # ✅ ДОБАВЛЕНО: Статистика людей возле поезда
    st.subheader("🚉 Люди возле поезда")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Максимум людей возле поезда", max_people_near_train)
        
    with col2:
        # Находим последний распознанный номер поезда
        last_train_number = None
        for event in reversed(train_events):
            if event.get("Номер"):
                last_train_number = event["Номер"]
                break
                
        if last_train_number:
            st.success(f"Последний распознанный поезд: {last_train_number}")

    # Визуализация распределения времени ожидания
    st.subheader("📈 Распределение времени ожидания")

    if total_people > 0:
        distribution_data = {
            "Категория": ["< 1 мин", "1-5 мин", "5-15 мин", "> 15 мин"],
            "Количество людей": [
                waiting_distribution["менее_1_мин"],
                waiting_distribution["1_5_мин"],
                waiting_distribution["5_15_мин"],
                waiting_distribution["более_15_мин"]
            ]
        }
        
        df_distribution = pd.DataFrame(distribution_data)
        st.bar_chart(df_distribution.set_index("Категория"))
        
        # Таблица с детализацией
        st.write("Детализация по категориям:")
        st.dataframe(df_distribution)
    else:
        st.info("Нет данных для анализа распределения")

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

    st.subheader("Вход/Выход")
    st.write(f"Вошло: {line.in_count} Вышло: {line.out_count}")

    # Дополнительно: кнопка для экспорта суммарной статистики
    if st.button("📥 Экспорт суммарной статистики ожидания"):
        summary_data = {
            "Общее время ожидания (мин)": total_waiting_time,
            "Всего людей": total_people,
            "Среднее время ожидания (мин)": total_waiting_time / total_people if total_people > 0 else 0,
            "Менее 1 мин": waiting_distribution["менее_1_мин"],
            "1-5 мин": waiting_distribution["1_5_мин"],
            "5-15 мин": waiting_distribution["5_15_мин"],
            "Более 15 мин": waiting_distribution["более_15_мин"],
            "Максимум людей возле поезда": max_people_near_train,
            "Последний номер поезда": last_train_number if last_train_number else "Не распознан"
        }
        
        df_summary = pd.DataFrame([summary_data])
        csv_summary = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать суммарную статистику (CSV)",
            data=csv_summary,
            file_name="platform_analysis_summary.csv",
            mime="text/csv"
        )

# --- Просмотр данных из базы ---
if db_enabled and os.path.exists(db_path):
    st.sidebar.header("Просмотр данных из БД")
    
    if st.sidebar.button("Показать историю анализов"):
        with get_db_connection() as conn:
            # Получаем список видео
            videos = conn.execute('''
                SELECT id, filename, processed_date, frame_count, processed_frames 
                FROM videos 
                ORDER BY processed_date DESC
            ''').fetchall()
            
            if videos:
                st.subheader("📋 История анализов видео")
                for video in videos:
                    with st.expander(f"{video['filename']} - {video['processed_date']}"):
                        st.write(f"ID: {video['id']}")
                        st.write(f"Кадров: {video['frame_count']} (обработано: {video['processed_frames']})")
                        
                        # Статистика по людям
                        people_stats = conn.execute(
                            'SELECT COUNT(*) as total_people FROM people WHERE video_id = ?', 
                            (video['id'],)
                        ).fetchone()
                        st.write(f"Всего людей: {people_stats['total_people']}")
                        
                        # Статистика по поездам
                        train_stats = conn.execute(
                            'SELECT COUNT(*) as total_trains FROM train_events WHERE video_id = ?', 
                            (video['id'],)
                        ).fetchone()
                        st.write(f"Событий поездов: {train_stats['total_trains']}")
                        
                        # Статистика времени ожидания
                        waiting_stats = conn.execute(
                            'SELECT total_waiting_minutes, average_waiting_minutes FROM waiting_statistics WHERE video_id = ?', 
                            (video['id'],)
                        ).fetchone()
                        if waiting_stats:
                            st.write(f"Общее время ожидания: {waiting_stats['total_waiting_minutes']:.1f} мин")
                            st.write(f"Среднее время ожидания: {waiting_stats['average_waiting_minutes']:.1f} мин")
                        
                        if st.button(f"Загрузить данные видео ID {video['id']}", key=f"load_{video['id']}"):
                            # Загружаем данные для этого видео
                            people_data_db = conn.execute(
                                'SELECT person_id, appearance_time, disappearance_time, waiting_minutes, reid_enabled FROM people WHERE video_id = ?',
                                (video['id'],)
                            ).fetchall()
                            
                            train_events_db = conn.execute(
                                'SELECT arrival_time, departure_time, duration_seconds, train_number FROM train_events WHERE video_id = ?',
                                (video['id'],)
                            ).fetchall()
                            
                            # Показываем данные
                            if people_data_db:
                                st.subheader("Люди из БД")
                                df_people_db = pd.DataFrame(people_data_db)
                                st.dataframe(df_people_db)
                            
                            if train_events_db:
                                st.subheader("Поезда из БД")
                                df_train_db = pd.DataFrame(train_events_db)
                                st.dataframe(df_train_db)
            else:
                st.info("В базе данных нет записей")