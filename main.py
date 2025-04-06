from video_processor import VideoProcessor
import os

# Configurações
VIDEO_PATH = "./Files/videos/Video2.mp4"  # Substitua pelo caminho do seu vídeo
MODEL_NAME = "./Models/license_plate_detection/yolov8n_plates/weights/best.pt"  # Use 'n' (mais rápido) até 'x' (mais preciso)
CONFIDENCE = 0.5  # Limiar de confiança (0.1 a 1.0)
SAVE_OUTPUT = True  # Salvar o vídeo processado
SHOW_DISPLAY = True  # Salvar o vídeo processado
OUTPUT_PATH = "./Output/VideosProcessados/"

# Configuração para salvar imagens dos objetos detectados
SAVE_OBJECT_CROPS = True      # Se True, salva recortes dos objetos detectados
CROPS_DIRECTORY = "./Output/Crops/"        # Se None, cria diretório automaticamente baseado no nome do vídeo
SAVE_INTERVAL = 0.5           # Intervalo mínimo em segundos entre salvamentos de placas

def main():
    """
    Função principal para processar o vídeo e detectar placas
    """
    # Verifica se o caminho do vídeo foi configurado
    if VIDEO_PATH == "seu_video.mp4":
        print("AVISO: Por favor, configure o caminho do vídeo na constante VIDEO_PATH!")
        video_path = input("Digite o caminho do vídeo: ")
    else:
        video_path = VIDEO_PATH
    
    # Verifica se o arquivo existe
    if not os.path.exists(video_path):
        print(f"Erro: O arquivo {video_path} não existe!")
        return
    
    # Mensagens informativas
    print(f"Iniciando detecção de placas de licença")
    if SAVE_OBJECT_CROPS:
        print(f"Salvando imagens com intervalo mínimo de {SAVE_INTERVAL} segundos")
    
    # Cria e configura o processador de vídeo
    processor = VideoProcessor(
        model_name=MODEL_NAME, 
        conf_threshold=CONFIDENCE,
        save_interval=SAVE_INTERVAL
    )
    
    # Processa o vídeo
    output_path, stats = processor.process_video(
        video_path=video_path,
        save_output=SAVE_OUTPUT,           # Salvar o vídeo processado
        display=SHOW_DISPLAY,             # Mostrar o vídeo durante o processamento
        save_crops=SAVE_OBJECT_CROPS,       # Salvar recortes das placas detectadas
        crops_dir=CROPS_DIRECTORY          # Diretório para salvar os recortes
    )
    
    # Exibe o resumo final
    print("\n" + "=" * 50)
    print("RESUMO DO PROCESSAMENTO")
    print("=" * 50)
    
    if SAVE_OUTPUT and output_path:
        print(f"Vídeo processado: {output_path}")
    
    if SAVE_OBJECT_CROPS:
        if CROPS_DIRECTORY:
            crops_path = CROPS_DIRECTORY
        else:
            video_filename = os.path.splitext(os.path.basename(video_path))[0]
            base_dir = os.path.dirname(video_path)
            crops_path = os.path.join(base_dir, f"{video_filename}_crops")
        
        total_crops = stats['saved_crops']
        print(f"Recortes salvos: {total_crops} imagens em {crops_path}")
        print(f"Intervalo mínimo entre salvamentos: {SAVE_INTERVAL} segundos")
        
        # Lista as classes detectadas
        if stats['detected_objects']:
            print("\nDetecções:")
            for cls_name, count in sorted(stats['detected_objects'].items(), key=lambda x: x[1], reverse=True):
                cls_dir = os.path.join(crops_path, cls_name)
                if os.path.exists(cls_dir):
                    num_files = len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
                    print(f"  - {cls_name}: {count} detecções, {num_files} imagens salvas")
                else:
                    print(f"  - {cls_name}: {count} detecções, 0 imagens salvas")
    
    total_detections = sum(stats['detected_objects'].values())
    print(f"\nTotal de detecções: {total_detections}")
    print(f"Tempo total de processamento: {stats['total_time']:.2f} segundos")
    print(f"FPS médio: {stats.get('avg_fps', 0):.2f}")

if __name__ == "__main__":
    main()