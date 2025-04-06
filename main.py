from video_processor import VideoProcessor

if __name__ == "__main__":
    # Configurações
    VIDEO_PATH = "./Files/videos/Video2.mp4"  # Substitua pelo caminho do seu vídeo
    MODEL_NAME = "./Models/license_plate_detection/yolov8n_plates/weights/best.pt"  # Use 'n' (mais rápido) até 'x' (mais preciso)
    CONFIDENCE = 0.5  # Limiar de confiança (0.1 a 1.0)
    SAVE_OUTPUT = True  # Salvar o vídeo processado
    SHOW_DISPLAY = True  # Salvar o vídeo processado
    OUTPUT_PATH = "./Output/VideosProcessados"
    
    processor = VideoProcessor(model_name=MODEL_NAME, conf_threshold=CONFIDENCE)
   
    output_path, stats = processor.process_video(
    video_path=VIDEO_PATH,
    save_output=SAVE_OUTPUT,   # Salvar o vídeo processado
    display=SHOW_DISPLAY,      # Mostrar o vídeo durante o processamento
    output_path =OUTPUT_PATH
)