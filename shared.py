import threading

# Global locks and frames
lock = threading.Lock()
stop_event = threading.Event()
latest_frame = None

# Holds a new camera URL if the user updates it from the web GUI
requested_camera_url = None
# You can also move system-wide configs here later (Phase 2)
save_to_sd = False
final_logs = []
image_cache = {}
available_classes = {}  # Holds the full dictionary from labels.txt
allowed_classes = []    # Holds the list of IDs the user wants to track
