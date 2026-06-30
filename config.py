MODEL_DETECTOR = 'traffic_model_edgetpu.tflite'
MODEL_CLASSIFIER = 'traffic_classifier_quant_edgetpu.tflite'
LABELS_FILE = 'labels.txt'
VIDEO_DEVICE = '/dev/video1'

TRACKER_MAX_DISTANCE = 300
TRACKER_MAX_FRAMES = 30
TRACKER_MAX_MISSING = 40
CROP_PADDING_RATIO = 0.35  # The 25% margin boost
