MODEL_DETECTOR = 'traffic_model_edgetpu.tflite'
MODEL_CLASSIFIER = 'traffic_classifier_quant_edgetpu.tflite'
LABELS_FILE = 'labels.txt'
VIDEO_DEVICE = 'rtsp://admin:fordsyke1944@192.168.1.119:554/ch0_0.264#media=video'

TRACKER_MAX_DISTANCE = 300
TRACKER_MAX_FRAMES = 30
TRACKER_MAX_MISSING = 40
CROP_PADDING_RATIO = 0.35  # The 25% margin boost
