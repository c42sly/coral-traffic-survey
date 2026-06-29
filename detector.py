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
    def __init__(self, device_path="/dev/video1"):
        self.cmd = ['v4l2-ctl', f'--device={device_path}', '--stream-mmap', '--stream-to=-']
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.frame = None
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
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

    def read(self): return self.frame
    def stop(self):
        self.stopped = True
        if self.proc: self.proc.kill()

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
    interpreter = make_interpreter('traffic_model_edgetpu.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    print("Opening direct kernel stream...")
    vs = VideoStream(device_path="/dev/video1").start()
    time.sleep(2.0)

    # Tracker uses max_missing=15 so it doesn't double-count cars that glitch for a few frames
    active_tracker = tracker.SmartBufferTracker(max_distance=250, max_frames=20, max_missing=15)

    try:
        while True:
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
                    xmin, ymin, xmax, ymax = map_bbox_to_frame(obj.bbox, scale_x, scale_y, pad_x, pad_y, frame_w, frame_h)
                    cv2.rectangle(draw_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    if xmax > xmin and ymax > ymin:
                        # --- THE CORRECTED 15% MARGIN LOGIC ---
                        w = xmax - xmin
                        h = ymax - ymin
                        pad_w = int(w * 0.25)
                        pad_h = int(h * 0.25)
                        
                        crop_xmin = max(0, xmin - pad_w)
                        crop_ymin = max(0, ymin - pad_h)
                        crop_xmax = min(frame_w, xmax + pad_w)
                        crop_ymax = min(frame_h, ymax + pad_h)
                        
                        crop_img = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()
                        # --------------------------------------
                        
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
                    "crops": vehicle.crops
                })

            with shared.lock:
                shared.latest_frame = draw_frame

    finally:
        vs.stop()
