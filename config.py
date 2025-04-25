"""
Centralized configuration using environment variables (.env)
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Paths and model
VIDEO_PATH = os.getenv('VIDEO_PATH', './Files/videos/Video2.mp4')
MODEL_NAME = os.getenv('MODEL_NAME', './Models/license_plate_detection/yolov8n_plates/weights/best.pt')

# Detection parameters
CONFIDENCE = float(os.getenv('CONFIDENCE', '0.5'))
SAVE_INTERVAL = float(os.getenv('SAVE_INTERVAL', '0.5'))

# Video output
SAVE_OUTPUT = os.getenv('SAVE_OUTPUT', 'True').lower() in ('true', '1', 'yes')
SHOW_DISPLAY = os.getenv('SHOW_DISPLAY', 'True').lower() in ('true', '1', 'yes')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './Output/VideosProcessados/')

# Crops
SAVE_OBJECT_CROPS = os.getenv('SAVE_OBJECT_CROPS', 'True').lower() in ('true', '1', 'yes')
CROPS_DIRECTORY = os.getenv('CROPS_DIRECTORY', './Output/Crops/')

# OCR
SAVE_OCR_RESULTS = os.getenv('SAVE_OCR_RESULTS', 'True').lower() in ('true', '1', 'yes')
import platform
TESSERACT_PATH = os.getenv('TESSERACT_PATH')
if not TESSERACT_PATH:
    if platform.system() == 'Windows':
        TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        TESSERACT_PATH = '/usr/bin/tesseract'

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
