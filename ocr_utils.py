import cv2
import pytesseract
import re
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OcrProcessor:
    """
    Classe para pré-processamento e reconhecimento OCR de placas.
    """
    def __init__(self, tesseract_cmd=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        # Pré-compile configurações de OCR e kernel
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.configs = [f'--oem 3 --psm {p} -c tessedit_char_whitelist={chars}' for p in (7,8,6)]
        self.clean_re = re.compile(f'[^{chars}]')
        self.kernel = np.ones((3,3), np.uint8)

    def preprocess(self, image):
        """
        Retorna três versões pré-processadas da imagem para OCR.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = max(1, 600 / max(gray.shape))
        if scale > 1:
            gray = cv2.resize(gray,
                              (int(gray.shape[1]*scale), int(gray.shape[0]*scale)),
                              interpolation=cv2.INTER_CUBIC)
        # versão 1
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        binar = cv2.adaptiveThreshold(blur, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        v1 = cv2.dilate(cv2.morphologyEx(binar, cv2.MORPH_OPEN, self.kernel), self.kernel)
        # versão 2
        eq = cv2.equalizeHist(gray)
        _, v2 = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # versão 3
        _, v3 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return v1, v2, v3

    def recognize(self, image):
        """
        Executa OCR com múltiplas configurações e retorna texto e melhor confiança.
        """
        best_text, best_conf = '', 0
        for cfg in self.configs:
            data = pytesseract.image_to_data(image, config=cfg, output_type=pytesseract.Output.DICT)
            parts, s, c = [], 0.0, 0
            for txt, conf in zip(data['text'], data['conf']):
                if txt.strip():
                    parts.append(txt)
                    # converte confiança para float com segurança
                    try:
                        conf_value = float(conf)
                    except (ValueError, TypeError):
                        continue
                    if conf_value > 0:
                        s += conf_value
                        c += 1
            text = self.clean_re.sub('', ''.join(parts))
            avg = s/c if c else 0
            if len(text)>len(best_text) or (len(text)==len(best_text) and avg>best_conf):
                best_text, best_conf = text, avg
        if not best_text:
            txt = self.clean_re.sub('', pytesseract.image_to_string(image, config=self.configs[0]).strip())
            if txt:
                best_text, best_conf = txt, 50
        #logger.debug(f'OCR result: {best_text} conf {best_conf}')
        return best_text, best_conf
