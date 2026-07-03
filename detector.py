import config
from datatypes import Detection
import cv2
import time
import threading
import subprocess
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

import tracker 
import shared

# --- Video Stream Class ---
class VideoStream:
    def __init__(self, device_path=config.VIDEO_DEVICE):
        self.device_path = str(device_path)
        # If it's a local USB camera, use v4l2. Otherwise, use OpenCV for network streams (RTSP/HTTP).
        self.is_v4l2 = self.device_path.startswith('/dev/video')
        self.frame = None
        self.stopped = False
        self.proc = None
        self.cap = None

        if self.is_v4l2:
            self.cmd = ['v4l2-ctl', f'--device={self.device_path}', '--stream-mmap', '--stream-to=-']
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        else:
            self.cap = cv2.VideoCapture(self.device_path)

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        if self.is_v4l2:
            JPEG_START, JPEG_END = b'\xff\xd8', b'\xff\xd9'
            buffer = b''
            while not self.stopped:
                chunk = self.proc.stdout.read(4096)
                if not chunk:
                    time.sleep(0.01)
                    continue
                buffer += chunk
                start_idx = buffer.find(JPEG_START)
                end_idx = buffer.find(JPEG_END, start_idx) if start_idx != -1 else -1
                if start_idx != -1 and end_idx != -1:
                    jpg_data = buffer[start_idx:end_idx + 2]
                    buffer = buffer[end_idx + 2:]
                    np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
                    decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if decoded is not None: self.frame = decoded
        else:
            # RTSP processing loop (drains the buffer to prevent lagging behind live time)
            while not self.stopped:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame = frame
                    else:
                        time.sleep(0.05)
                else:
                    time.sleep(0.1)

    def read(self): return self.frame
    def stop(self):
        self.stopped = True
        if self.proc: self.proc.kill()
        if self.cap: self.cap.release()

# --- Preprocessing helpers ---
def preprocess_letterbox(frame, input_width, input_height):
    frame_h, frame_w = frame.shape[:2]
    scale = min(input_width / frame_w, input_height / frame_h)
    new_w = int(round(frame_w * scale))
    new_h = int(round(frame_h * scale))
    resized = cv2.resize(frame, (new_w, new_h))
    pad_x = (input_width - new_w) // 2
    pad_y = (input_height - new_h) // 2
    canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    input_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return input_frame, (1.0 / scale), (1.0 / scale), pad_x, pad_y

def map_bbox_to_frame(bbox, scale_x, scale_y, pad_x, pad_y, frame_w, frame_h):
    xmin = int(max(0, min((bbox.xmin - pad_x) * scale_x, frame_w - 1)))
    xmax = int(max(0, min((bbox.xmax - pad_x) * scale_x, frame_w - 1)))
    ymin = int(max(0, min((bbox.ymin - pad_y) * scale_y, frame_h - 1)))
    ymax = int(max(0, min((bbox.ymax - pad_y) * scale_y, frame_h - 1)))
    return xmin, ymin, xmax, ymax

# --- Main Inference Loop ---
def inference_loop():
    interpreter = make_interpreter(config.MODEL_DETECTOR)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    
    current_camera = config.VIDEO_DEVICE
    print(f"Opening video stream on {current_camera}...")
    vs = VideoStream(device_path=current_camera).start()
    time.sleep(2.0)
    
    active_tracker = tracker.SmartBufferTracker(
        max_distance=config.TRACKER_MAX_DISTANCE, 
        max_frames=config.TRACKER_MAX_FRAMES, 
        max_missing=config.TRACKER_MAX_MISSING
    )

    try:
        while True:
            # --- NEW: Check if camera hot-swap was requested from the Web GUI ---
            with shared.lock:
                if hasattr(shared, 'requested_camera_url') and shared.requested_camera_url:
                    print(f"Hot-swapping camera to: {shared.requested_camera_url}")
                    vs.stop()
                    current_camera = shared.requested_camera_url
                    shared.requested_camera_url = None
                    vs = VideoStream(device_path=current_camera).start()
                    time.sleep(1.5) # Give the new stream time to connect
                    continue # Skip to next loop iteration

            # --- NEW: Pause Logic ---
            if not getattr(shared, 'is_detecting', False):
                # Drain the stream buffer to prevent lag, but skip inference
                if hasattr(vs, 'read'):
                    vs.read() 
                time.sleep(0.1) # Idle the CPU
                continue

            frame = vs.read()
            if frame is None: continue

            frame_h, frame_w = frame.shape[:2]
            input_frame, scale_x, scale_y, pad_x, pad_y = preprocess_letterbox(frame, input_width, input_height)

            common.set_input(interpreter, input_frame)
            interpreter.invoke()
            objs = detect.get_objects(interpreter, score_threshold=0.5)

            current_detections = []
            draw_frame = frame.copy()

            if objs:
                for obj in objs:
                    # --- NEW: The Class Filter Bouncer ---
                    if hasattr(shared, 'allowed_classes') and int(obj.id) not in shared.allowed_classes:
                        continue  # Skip this object immediately!

                    xmin, ymin, xmax, ymax = map_bbox_to_frame(obj.bbox, scale_x, scale_y, pad_x, pad_y, frame_w, frame_h)
                    cv2.rectangle(draw_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    if xmax > xmin and ymax > ymin:
                        w = xmax - xmin
                        h = ymax - ymin
                        pad_w = int(w * config.CROP_PADDING_RATIO)
                        pad_h = int(h * config.CROP_PADDING_RATIO)
                        
                        crop_xmin = max(0, xmin - pad_w)
                        crop_ymin = max(0, ymin - pad_h)
                        crop_xmax = min(frame_w, xmax + pad_w)
                        crop_ymax = min(frame_h, ymax + pad_h)
                        
                        crop_img = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()
                        
                        det = Detection(
                            bbox=(xmin, ymin, xmax, ymax),
                            crop=crop_img,
                            score=float(obj.score),
                            detector_label=int(obj.id)
                        )
                        current_detections.append(det)

            completed_vehicles = active_tracker.update(current_detections, time.time())

            from queue_manager import classifier_batch_queue
            for vehicle in completed_vehicles:
                classifier_batch_queue.put({
                    "track_id": vehicle.track_id,
                    "crops": vehicle.crops,
                    "detector_labels": vehicle.detector_labels,
                })

            with shared.lock:
                shared.latest_frame = draw_frame

    finally:
        vs.stop()
