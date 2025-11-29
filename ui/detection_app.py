import gradio as gr
from models.detection.factory_detector import simple_people_detection  # Импорт вашего кода

def run_detection(video):
    # Здесь вызовите simple_people_detection или process_video
    # Верните обработанное видео или изображение
    return "output_with_detections.avi"  # Или отобразите кадры

iface = gr.Interface(fn=run_detection, inputs="video", outputs="video")
if __name__ == "__main__":
    iface.launch()