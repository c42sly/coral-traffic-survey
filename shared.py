import threading

# Global locks and frames
lock = threading.Lock()
latest_frame = None

# Holds a new camera URL if the user updates it from the web GUI
requested_camera_url = None
# You can also move system-wide configs here later (Phase 2)
