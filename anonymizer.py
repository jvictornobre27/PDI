import cv2
import numpy as np
import time
from math import hypot
from ultralytics import YOLO

# Função de callback vazia necessária para os trackbars do OpenCV
def nada(x):
    pass

class FaceAnonymizer:
    def __init__(self, model_path="yolov8n-pose.pt", skip_frames=2):
        self.model = YOLO(model_path)
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.last_rois = [] 

        # Mapeamento das conexões do corpo (Pescoço e Membros)
        # Ignoramos intencionalmente os pontos 0 a 4 (Rosto) para manter a privacidade
        self.skeleton_bones = [
            (5, 6),   # Ombros
            (5, 7), (7, 9),   # Braço Esquerdo
            (6, 8), (8, 10),  # Braço Direito
            (5, 11), (6, 12), # Tronco
            (11, 12),         # Cintura
            (11, 13), (13, 15), # Perna Esquerda
            (12, 14), (14, 16)  # Perna Direita
        ]

    def calculate_head_roi(self, keypoints, frame_shape, conf_thresh):
        h_frame, w_frame = frame_shape[:2]
        
        nose = keypoints[0]
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        
        center_x, center_y = 0, 0
        roi_size = 0
        
        # Verifica individualmente o que está visível
        has_l_shoulder = l_shoulder[2] > conf_thresh
        has_r_shoulder = r_shoulder[2] > conf_thresh
        has_nose = nose[2] > conf_thresh
        
        # Cenário 1: Visão Frontal ou Traseira (Ambos os ombros visíveis)
        if has_l_shoulder and has_r_shoulder:
            shoulder_dist = hypot(r_shoulder[0] - l_shoulder[0], r_shoulder[1] - l_shoulder[1])
            roi_size = int(shoulder_dist * 1.5) 
            
            if has_nose:
                center_x, center_y = int(nose[0]), int(nose[1])
            else:
                # Projeção da pessoa de costas
                mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
                mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
                center_x = int(mid_x)
                # Ajuste 2: Reduzido de 1.2 para 0.9 (Evita que a caixa voe acima da cabeça)
                center_y = int(mid_y - (shoulder_dist * 0.9)) 
                
        # Cenário 2: Visão de Perfil (Nariz e apenas 1 ombro visível)
        elif has_nose and (has_l_shoulder or has_r_shoulder):
            center_x, center_y = int(nose[0]), int(nose[1])
            
            # Descobre qual ombro está visível e usa a distância até ele para escalar a caixa
            visible_shoulder = l_shoulder if has_l_shoulder else r_shoulder
            nose_to_shoulder_dist = hypot(nose[0] - visible_shoulder[0], nose[1] - visible_shoulder[1])
            
            roi_size = int(nose_to_shoulder_dist * 0.8) 
            
        else:
            # Perdeu as âncoras principais, aborta o frame
            return None 

        if roi_size == 0:
            return None

        # Limite mínimo para garantir que a caixa não desapareça na distância
        roi_size = max(roi_size, 20) 

        half_size = roi_size // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(w_frame, center_x + half_size)
        y2 = min(h_frame, center_y + half_size)
        
        return (x1, y1, x2, y2)

    def apply_filter(self, frame, roi_coords, filter_type, intensity):
        """ Aplica Pixelização ou Blur com base no seletor da UI. """
        x1, y1, x2, y2 = roi_coords
        
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return frame

        roi = frame[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        
        # Garante que a intensidade nunca seja 0
        intensity = max(1, intensity)
        
        if filter_type == 0:
            # Modo 0: Pixelização por subamostragem
            w_temp, h_temp = max(1, w // intensity), max(1, h // intensity)
            temp = cv2.resize(roi, (w_temp, h_temp), interpolation=cv2.INTER_LINEAR)
            filtered_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            # Modo 1: Gaussian Blur
            # O kernel do filtro Gaussiano precisa ser ímpar
            k_size = intensity if intensity % 2 != 0 else intensity + 1
            if k_size < 3: k_size = 3 # Kernel mínimo
            filtered_roi = cv2.GaussianBlur(roi, (k_size, k_size), 0)
            
        frame[y1:y2, x1:x2] = filtered_roi
        return frame

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        # Setup da janela de controle interativa
        window_name = "Painel Interativo"
        cv2.namedWindow(window_name)
        
        # Restauradas todas as opções + Novo seletor de Filtro
        cv2.createTrackbar("Filtro (0=Pix, 1=Blur)", window_name, 0, 1, nada)
        cv2.createTrackbar("Intensidade", window_name, 12, 51, nada) # Serve para Pixel ou Blur
        cv2.createTrackbar("Confianca (%)", window_name, 50, 100, nada)
        cv2.createTrackbar("Debug (0=Off, 1=On)", window_name, 1, 1, nada)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (854, 480))
            
            # Configuração dos trackbars
            ui_filter_type = cv2.getTrackbarPos("Filtro (0=Pix, 1=Blur)", window_name)
            ui_intensity = cv2.getTrackbarPos("Intensidade", window_name)
            ui_conf_thresh = cv2.getTrackbarPos("Confianca (%)", window_name) / 100.0
            ui_debug = cv2.getTrackbarPos("Debug (0=Off, 1=On)", window_name)

            start_time = time.perf_counter()
            t_ia_start = time.perf_counter()

            # Módulo de IA
            if self.frame_counter % self.skip_frames == 0:
                # Aumentando a resolução melhora a visão de longe, mas gasta mais processamento
                results = self.model(frame, imgsz=480, verbose=False)[0]
                current_rois = []
                
                if results.keypoints is not None and len(results.keypoints.data) > 0:
                    for kp in results.keypoints.data:
                        roi = self.calculate_head_roi(kp, frame.shape, ui_conf_thresh)
                        if roi:
                            current_rois.append((roi, kp))
                            
                self.last_rois = current_rois
            
            t_ia_end = time.perf_counter()

            # Módulo de PDI
            t_pdi_start = time.perf_counter()
            for roi_data in self.last_rois:
                roi_coords, kp = roi_data
                
                # Passa o tipo de filtro e a intensidade dinamicamente
                frame = self.apply_filter(frame, roi_coords, ui_filter_type, ui_intensity)
                
                if ui_debug == 1:
                    x1, y1, x2, y2 = roi_coords
                    # Desenha a caixa do rosto
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Desenha as "Juntas" (Pontos)
                    for i, point in enumerate(kp):
                        # Ignora os pontos do rosto (0 a 4) para garantir privacidade
                        if i > 4 and point[2] > ui_conf_thresh:
                            x, y = int(point[0]), int(point[1])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # Bolinhas verdes
                            
                    # 2. Desenha os "Ossos" do esqueleto (Linhas conectando as juntas)
                    for bone in self.skeleton_bones:
                        p1_idx, p2_idx = bone
                        p1, p2 = kp[p1_idx], kp[p2_idx]
                        
                        # Só desenha o esqueleto se o modelo de IA tiver confiança em ambas
                        if p1[2] > ui_conf_thresh and p2[2] > ui_conf_thresh:
                            pt1 = (int(p1[0]), int(p1[1]))
                            pt2 = (int(p2[0]), int(p2[1]))
                            cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
            
            t_pdi_end = time.perf_counter()

            # Métricas de desempenho
            total_time = time.perf_counter() - start_time
            fps = 1 / total_time if total_time > 0 else 0
            ia_latency = (t_ia_end - t_ia_start) * 1000
            pdi_latency = (t_pdi_end - t_pdi_start) * 1000

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"IA Latency: {ia_latency:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"PDI Latency: {pdi_latency:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Skip: {self.skip_frames} | Engine: YOLO-Pose", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow(window_name, frame)
            self.frame_counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceAnonymizer(model_path="yolov8n-pose.pt", skip_frames=2)
    app.process_video("sample.mp4")