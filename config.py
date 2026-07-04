MODEL_DETECTOR = 'traffic_model_edgetpu.tflite'
MODEL_CLASSIFIER = 'traffic_classifier_quant_edgetpu.tflite'
LABELS_FILE = 'labels.txt'
VIDEO_DEVICE = 'rtsp://admin:fordsyke1944@192.168.1.119:554/ch0_0.264#media=video'

TRACKER_MAX_DISTANCE = 500   #####Pixels
TRACKER_MAX_FRAMES = 120
TRACKER_MAX_MISSING = 60
CROP_PADDING_RATIO = 0.35  # The 35% margin boost
SAVE_DIR = "/mnt/server_output/diagnostic_crops"
