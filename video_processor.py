import cv2
from ultralytics import YOLO
import time
import os
import pytesseract
import re
import numpy as np

class VideoProcessor:
    """
    Classe para processar vídeos e detectar objetos usando YOLOv8
    """
    
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.25, save_interval=0.5, tesseract_cmd=None):
        """
        Inicializa o processador de vídeo
        
        Args:
            model_name (str): Nome do modelo YOLO a ser usado
            conf_threshold (float): Limiar de confiança para detecções (0.1 a 1.0)
            save_interval (float): Intervalo mínimo em segundos entre salvamentos de um mesmo tipo de objeto
            tesseract_cmd (str): Caminho para o executável do Tesseract OCR (se None, usa o padrão)
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.save_interval = save_interval
        self.model = None
        
        # Configuração do Tesseract OCR
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        self.processing_stats = {
            'frames_processed': 0,
            'total_time': 0,
            'processing_times': [],
            'detected_objects': {},
            'saved_crops': 0,
            'ocr_results': {}  # Armazena resultados do OCR por timestamp
        }
        # Dicionário para rastrear o último timestamp em que cada classe foi salva
        self.last_saved_time = {}
        
        # Lista para armazenar os resultados do OCR com seus metadados
        self.ocr_results_list = []
    
    def load_model(self):
        """
        Carrega o modelo YOLO
        
        Returns:
            bool: True se o modelo foi carregado com sucesso, False caso contrário
        """
        try:
            print(f"Carregando modelo {self.model_name}...")
            self.model = YOLO(self.model_name)
            print("Modelo carregado com sucesso!")
            return True
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            return False
    
    def process_video(self, video_path, save_output=True, display=True, output_path=None, save_crops=False, crops_dir=None):
        """
        Processa um vídeo para detectar placas de licença
        
        Args:
            video_path (str): Caminho para o arquivo de vídeo
            save_output (bool): Se True, salva o vídeo processado
            display (bool): Se True, exibe o vídeo durante o processamento
            output_path (str, optional): Caminho personalizado para o vídeo de saída
                                        Se None, gera um nome baseado no vídeo original
            save_crops (bool): Se True, salva imagens recortadas das placas detectadas
            crops_dir (str, optional): Diretório para salvar os recortes
                                     Se None, cria um diretório baseado no nome do vídeo
        
        Returns:
            str: Caminho para o vídeo de saída, se save_output=True
            dict: Estatísticas do processamento
        """
        # Reseta as estatísticas
        self.reset_stats()
        
        # Verifica se o arquivo existe
        if not os.path.exists(video_path):
            print(f"Erro: O arquivo {video_path} não existe!")
            return None, self.processing_stats
        
        # Carrega o modelo se ainda não foi carregado
        if self.model is None:
            if not self.load_model():
                return None, self.processing_stats
        
        # Abre o vídeo
        print(f"Abrindo vídeo: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Erro ao abrir o vídeo!")
            return None, self.processing_stats
        
        # Obtém informações do vídeo
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Resolução: {width}x{height}, FPS: {fps}, Total de frames: {total_frames}")
        
        # Configura o VideoWriter para salvar o resultado
        final_output_path = None
        out = None
        
        if save_output:
            if output_path is None:
                # Cria nome do arquivo de saída baseado no original
                filename, ext = os.path.splitext(os.path.basename(video_path))
                output_dir = os.path.dirname(video_path)
                final_output_path = os.path.join(output_dir, f"{filename}_detected{ext}")
            else:
                final_output_path = output_path
            
            # Define o codec e cria o VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Ou 'XVID' dependendo do sistema
            out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
            print(f"Salvando resultado em: {final_output_path}")
        
        # Configura diretório para salvar os recortes dos objetos
        if save_crops:
            if crops_dir is None:
                # Cria diretório baseado no nome do vídeo
                video_filename = os.path.splitext(os.path.basename(video_path))[0]
                base_dir = os.path.dirname(video_path)
                crops_dir = os.path.join(base_dir, f"{video_filename}_crops")
            
            # Cria o diretório se não existir
            os.makedirs(crops_dir, exist_ok=True)
            print(f"Salvando recortes dos objetos em: {crops_dir}")
        
        # Inicializa contadores para estatísticas
        start_time = time.time()
        frame_number = 0
        
        # Processa o vídeo frame a frame
        print("Iniciando processamento do vídeo...")
        video_timestamp = 0.0  # Timestamp atual do vídeo em segundos
        while cap.isOpened():
            # Lê um frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            self.processing_stats['frames_processed'] += 1
            
            # Atualiza o timestamp do vídeo
            video_timestamp = frame_number / fps
            
            # Marca o tempo de início do processamento
            frame_start_time = time.time()
            
            # Realiza a detecção
            results = self.model(frame, conf=self.conf_threshold)
            
            # Atualiza estatísticas de objetos detectados e salva recortes
            self._update_detection_stats(results[0], frame, frame_number, video_timestamp, save_crops, crops_dir)
            
            # Desenha os resultados no frame
            annotated_frame = results[0].plot()
            
            # Calcula o tempo de processamento para este frame
            frame_time = time.time() - frame_start_time
            self.processing_stats['processing_times'].append(frame_time)
            
            # Calcula e exibe o FPS atual
            current_fps = 1 / frame_time if frame_time > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostra o progresso
            if self.processing_stats['frames_processed'] % 10 == 0 or self.processing_stats['frames_processed'] == 1:
                elapsed_time = time.time() - start_time
                estimated_total = (elapsed_time / self.processing_stats['frames_processed']) * total_frames if self.processing_stats['frames_processed'] > 0 else 0
                remaining_time = estimated_total - elapsed_time
                
                print(f"Processando frame {self.processing_stats['frames_processed']}/{total_frames} " +
                     f"({(self.processing_stats['frames_processed']/total_frames*100):.1f}%) - " +
                     f"Tempo restante estimado: {remaining_time:.1f}s")
            
            # Mostra o frame
            if display:
                cv2.imshow("Detecção em vídeo", annotated_frame)
            
            # Salva o frame no vídeo de saída
            if out is not None:
                out.write(annotated_frame)
            
            # Verifica se o usuário pressionou 'q' para sair
            if display and (cv2.waitKey(1) & 0xFF == ord('q')):
                print("Processamento interrompido pelo usuário!")
                break
        
        # Calcula estatísticas finais
        self.processing_stats['total_time'] = time.time() - start_time
        
        # Libera recursos
        cap.release()
        if out is not None:
            out.release()
        
        if display:
            cv2.destroyAllWindows()
        
        # Exibe estatísticas finais
        self.print_stats(total_frames)
        
        return final_output_path, self.processing_stats
    
    def _update_detection_stats(self, result, frame, frame_number, video_timestamp, save_crops=False, crops_dir=None):
        """
        Atualiza as estatísticas de objetos detectados e salva recortes
        
        Args:
            result: Resultado da detecção YOLO para um frame
            frame: Frame original do vídeo
            frame_number: Número do frame atual
            video_timestamp: Timestamp atual do vídeo em segundos
            save_crops: Se True, salva recortes dos objetos detectados
            crops_dir: Diretório para salvar os recortes
        """
        # Conta os objetos detectados por classe e salva recortes
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls.item())
            cls_name = result.names[cls_id]
            conf = float(box.conf.item())
            
            # Atualiza contagem de objetos
            if cls_name not in self.processing_stats['detected_objects']:
                self.processing_stats['detected_objects'][cls_name] = 0
            
            self.processing_stats['detected_objects'][cls_name] += 1
            
            # Salva recortes dos objetos detectados
            if save_crops and crops_dir:
                # Verifica o intervalo de tempo desde o último salvamento desta classe
                current_time = video_timestamp
                last_time = self.last_saved_time.get(cls_name, -self.save_interval)  # -interval para garantir que a primeira detecção seja salva
                
                # Salva apenas se passado o intervalo de tempo mínimo desde o último salvamento
                should_save = False
                if current_time - last_time >= self.save_interval:
                    should_save = True
                    self.last_saved_time[cls_name] = current_time
                    
                    # Log para depuração
                    print(f"Salvando {cls_name} no timestamp {current_time:.2f}s (último salvo: {last_time:.2f}s)")
                
                if should_save:
                    # Obtém as coordenadas da caixa delimitadora
                    x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]
                    
                    # Certifica-se de que as coordenadas estão dentro dos limites do frame
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Extrai o recorte do objeto
                    crop = frame[y1:y2, x1:x2]
                    
                    # Realiza o pré-processamento para OCR - agora retorna 3 versões
                    preprocessed1, preprocessed2, preprocessed3 = self._preprocess_for_ocr(crop)
                    
                    # Tenta OCR em cada versão pré-processada para maximizar a chance de sucesso
                    plate_text1, confidence1 = self._recognize_plate(preprocessed1)
                    plate_text2, confidence2 = self._recognize_plate(preprocessed2)
                    plate_text3, confidence3 = self._recognize_plate(preprocessed3)
                    
                    # Escolhe o resultado com a maior confiança ou o texto mais longo
                    best_plate_text = ""
                    best_confidence = 0
                    best_preprocessed = preprocessed1
                    
                    # Critério: texto mais longo ganha, ou maior confiança em caso de empate
                    candidates = [
                        (plate_text1, confidence1, preprocessed1),
                        (plate_text2, confidence2, preprocessed2),
                        (plate_text3, confidence3, preprocessed3)
                    ]
                    
                    for text, conf, img in candidates:
                        if len(text) > len(best_plate_text) or (len(text) == len(best_plate_text) and conf > best_confidence):
                            best_plate_text = text
                            best_confidence = conf
                            best_preprocessed = img
                    
                    # Cria diretório para a classe se não existir
                    class_dir = os.path.join(crops_dir, cls_name)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    # Adiciona o texto reconhecido ao nome do arquivo
                    time_str = f"{video_timestamp:.2f}".replace(".", "_")
                    plate_text_clean = best_plate_text.replace(" ", "").replace("\n", "")
                    
                    if plate_text_clean:
                        crop_filename = f"time{time_str}s_frame{frame_number:06d}_obj{i:03d}_{cls_name}_{conf:.2f}_OCR_{plate_text_clean}.jpg"
                    else:
                        crop_filename = f"time{time_str}s_frame{frame_number:06d}_obj{i:03d}_{cls_name}_{conf:.2f}.jpg"
                    
                    crop_path = os.path.join(class_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    
                    # Salva também a imagem pré-processada que deu o melhor resultado
                    preproc_filename = f"time{time_str}s_frame{frame_number:06d}_obj{i:03d}_{cls_name}_preprocessed.jpg"
                    preproc_path = os.path.join(class_dir, preproc_filename)
                    cv2.imwrite(preproc_path, best_preprocessed)
                    
                    # Armazena os resultados do OCR
                    ocr_result = {
                        'timestamp': current_time,
                        'frame': frame_number,
                        'class': cls_name,
                        'confidence': conf,
                        'ocr_text': best_plate_text,
                        'ocr_confidence': best_confidence,
                        'image_path': crop_path,
                        'coordinates': (x1, y1, x2, y2)
                    }
                    
                    self.ocr_results_list.append(ocr_result)
                    
                    # Atualiza contador de recortes salvos
                    self.processing_stats['saved_crops'] += 1
    
    def _preprocess_for_ocr(self, image):
        """
        Pré-processa a imagem para melhorar o reconhecimento de OCR
        
        Args:
            image: Imagem da placa recortada
            
        Returns:
            Imagem pré-processada pronta para OCR
        """
        try:
            # Converte para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Redimensiona a imagem (opcional, mas pode ajudar com placas pequenas)
            scale_factor = max(1, 600 / max(gray.shape[0], gray.shape[1]))
            if scale_factor > 1:
                width = int(gray.shape[1] * scale_factor)
                height = int(gray.shape[0] * scale_factor)
                gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # Versão 1: Pré-processamento tradicional
            # Aplicar blur para reduzir ruído
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Aplicar threshold adaptativo para binarizar a imagem
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Operações morfológicas para remover ruído
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Dilatar um pouco para conectar componentes próximos
            preprocessed1 = cv2.dilate(opening, kernel, iterations=1)
            
            # Versão 2: Equalização de histograma
            # Equaliza o histograma para melhorar o contraste
            equalized = cv2.equalizeHist(gray)
            
            # Aplicar Otsu's thresholding
            _, preprocessed2 = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Versão 3: Thresholding simples
            # Aplica uma simples binarização
            _, preprocessed3 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Retorna a versão 1 como padrão, mas tenta as outras versões no reconhecimento
            return preprocessed1, preprocessed2, preprocessed3
        except Exception as e:
            print(f"Erro no pré-processamento: {e}")
            return image, image, image  # Retorna a imagem original em caso de erro
    
    def _recognize_plate(self, preprocessed_image):
        """
        Reconhece o texto da placa usando OCR
        
        Args:
            preprocessed_image: Imagem pré-processada da placa
            
        Returns:
            texto: Texto reconhecido na placa
            confidence: Confiança da detecção
        """
        try:
            # Configurações para o Tesseract
            # PSM 7: Trata a imagem como uma única linha de texto
            # PSM 8: Trata a imagem como uma única palavra
            # PSM 6: Assume um único bloco de texto uniforme
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Tenta com várias configurações para aumentar as chances de reconhecimento
            configs = [
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]
            
            best_text = ""
            best_confidence = 0
            
            # Tenta diferentes configurações e escolhe o melhor resultado
            for config in configs:
                # Executa o OCR
                data = pytesseract.image_to_data(preprocessed_image, config=config, output_type=pytesseract.Output.DICT)
                
                # Extrai o texto e a confiança média
                text_parts = []
                confidence_sum = 0
                confidence_count = 0
                
                for i in range(len(data['text'])):
                    if data['text'][i].strip():
                        text_parts.append(data['text'][i])
                        confidence_sum += float(data['conf'][i]) if data['conf'][i] > 0 else 0
                        confidence_count += 1
                
                # Calcula a confiança média
                avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
                
                # Junta as partes do texto
                text = "".join(text_parts)
                
                # Limpeza adicional: remove caracteres não alfanuméricos
                text = re.sub(r'[^A-Z0-9]', '', text)
                
                # Se encontramos um texto melhor (mais longo ou com maior confiança)
                if len(text) > len(best_text) or (len(text) == len(best_text) and avg_confidence > best_confidence):
                    best_text = text
                    best_confidence = avg_confidence
            
            # Tenta também o modo simples quando os outros falham
            if not best_text:
                # Tenta um reconhecimento direto simples
                simple_text = pytesseract.image_to_string(
                    preprocessed_image, 
                    config=r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ).strip()
                
                simple_text = re.sub(r'[^A-Z0-9]', '', simple_text)
                if simple_text:
                    best_text = simple_text
                    best_confidence = 50.0  # Confiança padrão para reconhecimento simples
            
            print(f"OCR detectou: '{best_text}' com confiança {best_confidence:.1f}%")
            
            return best_text, best_confidence
        except Exception as e:
            print(f"Erro no OCR: {e}")
            return "", 0.0
    
    def reset_stats(self):
        """
        Reseta as estatísticas de processamento
        """
        self.processing_stats = {
            'frames_processed': 0,
            'total_time': 0,
            'processing_times': [],
            'detected_objects': {},
            'saved_crops': 0
        }
    
    def print_stats(self, total_frames=None):
        """
        Imprime as estatísticas de processamento
        
        Args:
            total_frames (int, optional): Total de frames no vídeo
        """
        # Calcula estatísticas derivadas
        avg_time_per_frame = sum(self.processing_stats['processing_times']) / len(self.processing_stats['processing_times']) if self.processing_stats['processing_times'] else 0
        avg_fps = 1 / avg_time_per_frame if avg_time_per_frame > 0 else 0
        
        # Exibe estatísticas finais
        print("\nEstatísticas de processamento:")
        if total_frames:
            print(f"Frames processados: {self.processing_stats['frames_processed']}/{total_frames}")
        else:
            print(f"Frames processados: {self.processing_stats['frames_processed']}")
            
        print(f"Tempo total: {self.processing_stats['total_time']:.2f} segundos")
        print(f"Tempo médio por frame: {avg_time_per_frame*1000:.2f} ms")
        print(f"FPS médio: {avg_fps:.2f}")
        
        # Exibe estatísticas de objetos detectados
        total_detections = sum(self.processing_stats['detected_objects'].values())
        print(f"\nTotal de detecções: {total_detections}")
        
        if self.processing_stats['detected_objects']:
            print("Objetos detectados:")
            for cls_name, count in sorted(self.processing_stats['detected_objects'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls_name}: {count}")
        
        # Exibe estatísticas de recortes salvos
        if self.processing_stats['saved_crops'] > 0:
            print(f"\nRecortes de objetos salvos: {self.processing_stats['saved_crops']}")
            
        # Exibe resultados do OCR
        if self.ocr_results_list:
            print("\nResultados do OCR:")
            for i, result in enumerate(self.ocr_results_list[:10], 1):  # Limita a 10 resultados para não sobrecarregar o console
                if result['ocr_text']:
                    print(f"  {i}. Placa: {result['ocr_text']} (Confiança: {result['ocr_confidence']:.1f}%) - Frame {result['frame']} @ {result['timestamp']:.2f}s")
            
            if len(self.ocr_results_list) > 10:
                print(f"  ... e mais {len(self.ocr_results_list) - 10} resultados")
    
    def save_ocr_results(self, output_file):
        """
        Salva os resultados do OCR em um arquivo CSV
        
        Args:
            output_file (str): Caminho para o arquivo de saída
        
        Returns:
            bool: True se salvou com sucesso, False caso contrário
        """
        if not self.ocr_results_list:
            print("Nenhum resultado de OCR para salvar")
            return False
            
        try:
            import csv
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'frame', 'class', 'confidence', 'ocr_text', 'ocr_confidence', 'image_path']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in self.ocr_results_list:
                    # Prepara uma versão simplificada para o CSV
                    row = {
                        'timestamp': result['timestamp'],
                        'frame': result['frame'],
                        'class': result['class'],
                        'confidence': result['confidence'],
                        'ocr_text': result['ocr_text'],
                        'ocr_confidence': result['ocr_confidence'],
                        'image_path': result['image_path']
                    }
                    writer.writerow(row)
                    
            print(f"Resultados de OCR salvos em: {output_file}")
            
            # Imprime alguns resultados para depuração
            print("\nAmostra de dados salvos no CSV:")
            for i, result in enumerate(self.ocr_results_list[:5]):
                print(f"  {i+1}. Frame: {result['frame']}, Texto: '{result['ocr_text']}', Confiança: {result['ocr_confidence']:.1f}%")
            
            return True
        except Exception as e:
            print(f"Erro ao salvar resultados de OCR: {e}")
            return False
    
    def get_stats(self):
        """
        Retorna as estatísticas de processamento
        
        Returns:
            dict: Estatísticas de processamento
        """
        # Adiciona estatísticas derivadas
        stats = self.processing_stats.copy()
        
        if stats['processing_times']:
            avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
            stats['avg_time_per_frame'] = avg_time
            stats['avg_fps'] = 1 / avg_time if avg_time > 0 else 0
        else:
            stats['avg_time_per_frame'] = 0
            stats['avg_fps'] = 0
            
        return stats