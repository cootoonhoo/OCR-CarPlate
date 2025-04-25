import os
from video_processor import VideoProcessor
from config import *
import logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Configurações

def main():
    """
    Função principal para processar o vídeo e detectar placas
    """
    # Verifica se o caminho do vídeo foi configurado
    video_path = VIDEO_PATH
    
    # Verifica se o arquivo existe
    if not os.path.exists(video_path):
        logger.error(f"Arquivo não encontrado: {video_path}")
        return
    
    # Verifica o caminho do Tesseract
    if not os.path.exists(TESSERACT_PATH):
        logger.warning(f"Tesseract não encontrado em {TESSERACT_PATH}. OCR pode falhar.")

    
    # Cria e configura o processador de vídeo
    processor = VideoProcessor(
        model_name=MODEL_NAME,
        conf_threshold=CONFIDENCE,
        save_interval=SAVE_INTERVAL,
        tesseract_cmd=TESSERACT_PATH
    )
    
    # Processa o vídeo
    output_path, stats = processor.process_video(
        video_path,
        save_output=SAVE_OUTPUT,
        display=SHOW_DISPLAY,
        save_crops=SAVE_OBJECT_CROPS,
        crops_dir=CROPS_DIRECTORY
    )
    
    # Define o diretório onde os recortes foram salvos
    if CROPS_DIRECTORY:
        crops_path = CROPS_DIRECTORY
    else:
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        base_dir = os.path.dirname(video_path)
        crops_path = os.path.join(base_dir, f"{video_filename}_crops")
    
    # Salva os resultados do OCR em um arquivo CSV
    if SAVE_OCR_RESULTS and processor.ocr_results_list:
        csv_path = os.path.join(CROPS_DIRECTORY, "ocr_results.csv")
        processor.save_ocr_results(csv_path)
    
    # Exibe o resumo final
    logger.info("=" * 50)
    logger.info("RESUMO DO PROCESSAMENTO")
    logger.info("=" * 50)
    
    if SAVE_OUTPUT and output_path:
        logger.info(f"Vídeo processado: {output_path}")
    
    if SAVE_OBJECT_CROPS:
        total_crops = stats['saved_crops']
        logger.info(f"Recortes salvos: {total_crops} imagens em {crops_path}")
        logger.info(f"Intervalo mínimo entre salvamentos: {SAVE_INTERVAL} segundos")
        
        # Lista as classes detectadas
        if stats['detected_objects']:
            logger.info("Detecções:")
            for cls_name, count in sorted(stats['detected_objects'].items(), key=lambda x: x[1], reverse=True):
                cls_dir = os.path.join(crops_path, cls_name)
                if os.path.exists(cls_dir):
                    num_files = len([f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))])
                    logger.info(f"  - {cls_name}: {count} detecções, {num_files} imagens salvas")
                else:
                    logger.info(f"  - {cls_name}: {count} detecções, 0 imagens salvas")
    
    # Resumo final
    total_detections = sum(stats['detected_objects'].values())
    summary = [
        f"Total de detecções: {total_detections}",
        f"Tempo total: {stats['total_time']:.2f}s",
        f"FPS médio: {stats.get('avg_fps', 0):.2f}"
    ]
    logger.info("\n" + "\n".join(summary))
    logger.info("=" * 50)
    logger.info("INSTRUÇÃO PARA VISUALIZAR OS RESULTADOS")
    logger.info("=" * 50)
    instructions = [
        f"1. Acesse a pasta: {crops_path}",
        "2. Resultado OCR: ocr_results.csv",  
        "3. Placa: original e pré-processada (_preprocessed)",
        "4. Nome contém texto OCR (prefixo 'OCR_')"
    ]
    for line in instructions:
        logger.info(line)

if __name__ == "__main__":
    main()