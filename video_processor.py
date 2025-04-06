import cv2
from ultralytics import YOLO
import time
import os

class VideoProcessor:
    """
    Classe para processar vídeos e detectar objetos usando YOLOv8
    """
    
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.25, save_interval=0.5):
        """
        Inicializa o processador de vídeo
        
        Args:
            model_name (str): Nome do modelo YOLO a ser usado
            conf_threshold (float): Limiar de confiança para detecções (0.1 a 1.0)
            save_interval (float): Intervalo mínimo em segundos entre salvamentos de um mesmo tipo de objeto
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.save_interval = save_interval
        self.model = None
        self.processing_stats = {
            'frames_processed': 0,
            'total_time': 0,
            'processing_times': [],
            'detected_objects': {},
            'saved_crops': 0
        }
        # Dicionário para rastrear o último timestamp em que cada classe foi salva
        self.last_saved_time = {}
    
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
    
    def process_video(self, video_path, save_output=True, display=True, output_path=None, save_crops=False, crops_dir=None, target_class="license_plate"):
        """
        Processa um vídeo para detectar objetos
        
        Args:
            video_path (str): Caminho para o arquivo de vídeo
            save_output (bool): Se True, salva o vídeo processado
            display (bool): Se True, exibe o vídeo durante o processamento
            output_path (str, optional): Caminho personalizado para o vídeo de saída
                                        Se None, gera um nome baseado no vídeo original
            save_crops (bool): Se True, salva imagens recortadas dos objetos detectados
            crops_dir (str, optional): Diretório para salvar os recortes
                                     Se None, cria um diretório baseado no nome do vídeo
            target_class (str): Classe alvo para detecção (por padrão "placa")
        
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
            self._update_detection_stats(results[0], frame, frame_number, video_timestamp, save_crops, crops_dir, target_class)
            
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
    
    def _update_detection_stats(self, result, frame, frame_number, video_timestamp, save_crops=False, crops_dir=None, target_class="placa"):
        """
        Atualiza as estatísticas de objetos detectados e salva recortes
        
        Args:
            result: Resultado da detecção YOLO para um frame
            frame: Frame original do vídeo
            frame_number: Número do frame atual
            video_timestamp: Timestamp atual do vídeo em segundos
            save_crops: Se True, salva recortes dos objetos detectados
            crops_dir: Diretório para salvar os recortes
            target_class: Classe alvo para detecção (por padrão "placa")
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
                # Verifica se a classe do objeto é a que estamos procurando (considerando diferentes nomenclaturas possíveis)
                # Verificação case-insensitive e parcial para capturar variações como "placa", "placas", "license_plate", etc.
                should_save = False
                
                # Verifique se a classe atual corresponde ao alvo (sem diferenciar maiúsculas/minúsculas)
                if (target_class.lower() in cls_name.lower() or 
                    cls_name.lower() in target_class.lower() or
                    cls_name.lower() == target_class.lower()):
                    
                    # Verifica o intervalo de tempo desde o último salvamento desta classe
                    current_time = video_timestamp
                    last_time = self.last_saved_time.get(cls_name, -self.save_interval)  # -interval para garantir que a primeira detecção seja salva
                    
                    # Salva apenas se passado o intervalo de tempo mínimo desde o último salvamento
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
                    
                    # Cria diretório para a classe se não existir
                    class_dir = os.path.join(crops_dir, cls_name)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    # Salva o recorte
                    time_str = f"{video_timestamp:.2f}".replace(".", "_")
                    crop_filename = f"time{time_str}s_frame{frame_number:06d}_obj{i:03d}_{cls_name}_{conf:.2f}.jpg"
                    crop_path = os.path.join(class_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    
                    # Atualiza contador de recortes salvos
                    self.processing_stats['saved_crops'] += 1
    
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