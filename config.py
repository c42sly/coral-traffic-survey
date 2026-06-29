MODEL_DETECTOR = 'traffic_model_edgetpu.tflite'
MODEL_CLASSIFIER = 'traffic_classifier_quant_edgetpu.tflite'
LABELS_FILE = 'labels.txt'
VIDEO_DEVICE = '/dev/video1'

TRACKER_MAX_DISTANCE = 250
TRACKER_MAX_FRAMES = 20
TRACKER_MAX_MISSING = 30
CROP_PADDING_RATIO = 0.35  # The 25% margin boost
