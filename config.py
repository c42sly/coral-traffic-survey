MODEL_DETECTOR = 'traffic_model_edgetpu.tflite'
MODEL_CLASSIFIER = 'traffic_classifier_quant_edgetpu.tflite'
LABELS_FILE = 'labels.txt'                  # Classifier labels — Keras alphabetical sort order
DETECTOR_LABELS_FILE = 'detector_labels.txt' # Detector labels — model training order

VIDEO_DEVICE = 'rtsp://admin:fordsyke1944@192.168.1.119:554/ch0_0.264#media=video'

# --- Detector label mapping ---
# Source: labelmap.txt embedded in traffic_model.tflite (extracted via unzip).
# This is a freshly trained model so the embedded labelmap is the authoritative
# source — previous empirical confirmations (id3=Ag trailer, id4=Ag vehicle)
# were from an older model and do not apply here.
#
# The detector's label guess is informational/diagnostic only — the classifier
# makes all final classification decisions. If det_guess doesn't match the
# classifier output, that's expected and interesting, not a bug.
#
# To verify any specific ID: set DEBUG_PRINT_DETECTIONS = True below and walk
# a single known vehicle type past the camera in isolation. Read the raw id
# directly off the console.
DETECTOR_LABELS = {
    0:  "car",
    1:  "person",
    2:  "bicycle",
    3:  "LGV",
    4:  "OGV1",
    5:  "OGV2",
    6:  "motorcycle",
    7:  "Agricultural vehicle",
    8:  "Agricultural trailer",
    9:  "agricultural implement",
    10: "bus",
    11: "Trailer",
}

# --- Tracker settings ---
# Tuned for mixed rural/A-road traffic. See comments for guidance on tuning
# TRACKER_MAX_DISTANCE if you move the camera to a new location.
TRACKER_MAX_MISSING = 12     # Frames before a lost track is finalised
TRACKER_MAX_FRAMES = 60      # Safety cap — max crops per track before forced finalisation
TRACKER_MAX_DISTANCE = 250   # Max centroid jump in pixels between frames.
                              # Too tight -> fast vehicles fragment into multiple short tracks
                              # (visible as 1-2 crop discards of clean vehicles).
                              # Too loose -> tracks jump between lanes.
                              # Calibrate by watching DISCARDED_CROPS_DIR on site.

CROP_PADDING_RATIO = 0.35    # Proportional padding added around each detected bbox
                              # before cropping and sending to the classifier.

# --- Output directories ---
SAVE_DIR = "/mnt/server_output/diagnostic_crops"
SAVE_DISCARDED_CROPS = True
DISCARDED_CROPS_DIR = "/mnt/server_output/discarded_crops"

# --- Detector confidence threshold ---
# Lower this if vehicles are being missed entirely (no track created, nothing
# in discarded crops). Motion blur at speed can push genuine detections below
# the default 0.5.
DETECTOR_SCORE_THRESHOLD = 0.5

# --- Debug flags ---
# Print every raw detector (id, score) pair as detections arrive.
# Useful for confirming DETECTOR_LABELS mapping: point camera at one known
# vehicle type at a time and read the id directly. Turn off on a busy road —
# it's very noisy.
DEBUG_PRINT_DETECTIONS = False
