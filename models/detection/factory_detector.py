import cv2
import numpy as np
from datetime import datetime
import os

class FactoryPeopleDetector:
    def __init__(self, model_type='yolo'):
        """
        Инициализация детектора людей
        model_type: 'yolo' - YOLO (рекомендуется), 'hog' - HOG (быстрее, но менее точно)
        """
        self.model_type = model_type
        
        if model_type == 'yolo':
            # Загрузка YOLO модели (более точная) — убедитесь, что yolov3.weights, yolov3.cfg и coco.names в data/
            self.net = cv2.dnn.readNet('data/yolov3.weights', 'data/yolov3.cfg')
            with open('data/coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            
        elif model_type == 'hog':
            # Инициализация HOG детектора (встроен в OpenCV)
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Статистика
        self.people_count = 0
        self.detection_log = []
    
    def detect_people_yolo(self, frame):
        """Обнаружение людей с помощью YOLO"""
        height, width = frame.shape[:2]
        
        # Подготовка изображения для YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Фильтрация только людей (class_id = 0 в COCO dataset)
                if class_id == 0 and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Координаты прямоугольника
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Применение Non-Maximum Suppression для устранения дублирующих детекций
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        people_detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                people_detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidences[i]
                })
        
        return people_detections
    
    def detect_people_hog(self, frame):
        """Обнаружение людей с помощью HOG"""
        # Изменение размера для улучшения производительности
        frame_resized = cv2.resize(frame, (640, 480))
        
        # Обнаружение людей
        boxes, weights = self.hog.detectMultiScale(
            frame_resized,
            winStride=(8, 8),
            padding=(32, 32),
            scale=1.05,
            hitThreshold=0.5
        )
        
        people_detections = []
        for i, (x, y, w, h) in enumerate(boxes):
            # Масштабирование координат обратно к исходному размеру
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)
            w_orig = int(w * scale_x)
            h_orig = int(h * scale_y)
            
            people_detections.append({
                'bbox': (x_orig, y_orig, w_orig, h_orig),
                'confidence': weights[i] if i < len(weights) else 0.5
            })
        
        return people_detections
    
    def draw_detections(self, frame, detections):
        """Отрисовка bounding box'ов и информации на кадре"""
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Рисование прямоугольника
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Подпись с уверенностью
            label = f'Person: {confidence:.2f}'
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Отображение общего количества людей
        cv2.putText(frame, f'People detected: {len(detections)}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def process_video(self, video_path, output_path=None, show_video=True):
        """
        Обработка видеофайла
        video_path: путь к входному видео
        output_path: путь для сохранения результата (опционально)
        show_video: показывать ли видео в реальном времени
        """
        cap = cv2.VideoCapture(video_path)
        
        # Настройка выходного видео
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_people = 0
        
        print("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Обнаружение людей
            if self.model_type == 'yolo':
                detections = self.detect_people_yolo(frame)
            else:
                detections = self.detect_people_hog(frame)
            
            # Логирование детекций
            if detections:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.detection_log.append({
                    'timestamp': timestamp,
                    'frame': frame_count,
                    'people_count': len(detections)
                })
                total_people += len(detections)
            
            # Отрисовка результатов
            frame_with_detections = self.draw_detections(frame.copy(), detections)
            
            # Сохранение результата
            if output_path:
                out.write(frame_with_detections)
            
            # Отображение
            if show_video:
                cv2.imshow('Factory People Detection', frame_with_detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Завершение работы
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Вывод статистики
        print(f"\nProcessing completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total people detected: {total_people}")
        print(f"Average people per frame: {total_people/frame_count:.2f}")
        
        return total_people

# Дополнительные утилиты
class FactoryAnalytics:
    def __init__(self, detector):
        self.detector = detector
    
    def generate_report(self):
        """Генерация отчета по обнаружению людей"""
        if not self.detector.detection_log:
            print("No detections to report")
            return
        
        print("\n" + "="*50)
        print("FACTORY PEOPLE DETECTION REPORT")
        print("="*50)
        
        # Анализ по часам
        hourly_counts = {}
        for detection in self.detector.detection_log:
            hour = detection['timestamp'].split(' ')[1].split(':')[0]
            hourly_counts[hour] = hourly_counts.get(hour, 0) + detection['people_count']
        
        print("\nHourly Activity:")
        for hour in sorted(hourly_counts.keys()):
            print(f"Hour {hour}:00 - {hourly_counts[hour]} people detected")
        
        # Сохранение лога в файл
        log_filename = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_filename, 'w') as f:
            f.write("Timestamp,Frame,PeopleCount\n")
            for detection in self.detector.detection_log:
                f.write(f"{detection['timestamp']},{detection['frame']},{detection['people_count']}\n")
        
        print(f"\nDetailed log saved to: {log_filename}")

# Простая функция (из первого фрагмента) — как утилита
def simple_people_detection(video_path):
    """
    Простая версия обнаружения людей на видео с завода (использует HOG)
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Изменение размера для производительности
        frame_resized = cv2.resize(frame, (640, 480))
        
        # Обнаружение людей
        boxes, weights = hog.detectMultiScale(
            frame_resized,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05
        )
        
        # Отрисовка bounding box'ов
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Отображение количества людей
        cv2.putText(frame_resized, f'People: {len(boxes)}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Factory Monitoring', frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Альтернативный вариант для работы с камерой в реальном времени
def real_time_detection(model_type='hog'):
    """Обнаружение людей в реальном времени с камеры"""
    detector = FactoryPeopleDetector(model_type=model_type)
    
    cap = cv2.VideoCapture(0)  # Веб-камера
    
    print("Starting real-time detection... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обнаружение людей
        if model_type == 'yolo':
            detections = detector.detect_people_yolo(frame)
        else:
            detections = detector.detect_people_hog(frame)
        
        # Отрисовка результатов
        frame_with_detections = detector.draw_detections(frame, detections)
        
        # Отображение
        cv2.imshow('Real-Time Factory Monitoring', frame_with_detections)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Пример использования
def main():
    # Инициализация детектора
    print("Initializing people detector...")
    
    # Выбор модели: 'yolo' для точности, 'hog' для скорости
    detector = FactoryPeopleDetector(model_type='hog')  # Используем HOG для демонстрации
    
    # Обработка видео
    video_file = "data/factory_video.mp4"  # Замените на путь к вашему видео в data/
    
    if os.path.exists(video_file):
        total_detections = detector.process_video(
            video_path=video_file,
            output_path="output_with_detections.avi",
            show_video=True
        )
        
        # Генерация отчета
        analytics = FactoryAnalytics(detector)
        analytics.generate_report()
    else:
        print(f"Video file {video_file} not found!")
        print("Please provide a valid video file path.")

if __name__ == "__main__":
    # Запуск обработки видеофайла
    main()
    
    # Или запуск реального времени (раскомментируйте следующую строку)
    # real_time_detection()