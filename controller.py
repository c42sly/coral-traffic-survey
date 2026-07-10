import threading
import shared, queue_manager, detector, classifier, time
import re
import config
import os
import cv2
import shutil
from flask import send_file

_threads = []

# Ensure our save directory exists
SAVE_DIR = getattr(config, 'SAVE_DIR', 'saved_crops')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)


def start():
    """Start detector, classifier, results threads and load labels."""
    load_labels()

    det_thread = threading.Thread(target=detector.inference_loop, daemon=True)
    det_thread.start()
    _threads.append(det_thread)

    class_thread = threading.Thread(target=classifier.classifier_worker, daemon=True)
    class_thread.start()
    _threads.append(class_thread)

    res_thread = threading.Thread(target=results_worker, daemon=True)
    res_thread.start()
    _threads.append(res_thread)

    print("All threads started.")


def stop():
    """Signal threads to stop and wait for them."""
    shared.running = False
    for t in _threads:
        t.join(timeout=1.0)
    print("Threads stopped.")


def toggle_save():
    shared.save_to_sd = not shared.save_to_sd
    return shared.save_to_sd


def set_camera_url(url):
    try:
        with open('config.py', 'r') as f:
            content = f.read()
        content_new = re.sub(r"^VIDEO_DEVICE = .*$", f"VIDEO_DEVICE = '{url}'",
                             content, flags=re.MULTILINE)
        with open('config.py', 'w') as f:
            f.write(content_new)
    except Exception:
        return False
    shared.requested_camera_url = url
    return True


def get_config_camera():
    try:
        return config.VIDEO_DEVICE
    except AttributeError:
        return ""


def get_sys_stats():
    stats = {"cpu_usage": 0, "cpu_temp": 0, "tpu_temp": 0,
             "ram_usage": 0, "power_watts": 0.0}
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            stats["cpu_temp"] = round(int(f.read().strip()) / 1000.0, 1)
    except: pass
    try:
        with open('/sys/class/apex/apex_0/temp', 'r') as f:
            stats["tpu_temp"] = round(int(f.read().strip()) / 1000.0, 1)
    except: pass
    try:
        load1, _, _ = os.getloadavg()
        stats["cpu_usage"] = round(min((load1 / 4.0) * 100, 100.0), 1)
    except: pass
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
            mem_total = int(lines[0].split()[1])
            mem_available = int(lines[2].split()[1])
            stats["ram_usage"] = round(
                ((mem_total - mem_available) / mem_total) * 100, 1)
    except: pass
    with shared.lock:
        stats["power_watts"] = getattr(shared, 'current_power_watts', 0.0)
    return stats


# --- Background Worker to Process Results ---
def results_worker():
    print("Background result writer started.")

    while True:
        result = queue_manager.results_queue.get()
        crop = result.pop("crop", None)

        result['timestamp'] = time.strftime("%H:%M:%S")

        # --- Classifier-side bouncer (case-insensitive name match) ---
        # available_classes is now keyed by detector ID with detector label
        # names as values. allowed_classes is the list of detector IDs the
        # user has enabled. We convert both to lowercase names and check
        # the classifier's output class_name against them.
        # This works because detector and classifier label names match
        # (same training dataset), with only minor capitalisation differences
        # handled by .lower().
        if hasattr(shared, 'allowed_classes') and hasattr(shared, 'available_classes'):
            allowed_names = [
                str(shared.available_classes.get(cid)).lower()
                for cid in shared.allowed_classes
            ]
            ai_guess = str(result.get('class_name', '')).lower()
            if ai_guess not in allowed_names:
                continue

        # 1. Save to SD Card (if toggled ON)
        if shared.save_to_sd and crop is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_class = result['class_name'].replace(' ', '')
            conf = int(result['score'] * 100)
            filename = (f"{timestamp}_vid{result['track_id']}"
                        f"_{safe_class}_conf{conf}.jpg")
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, crop)
            print(f"💾 Saved crop to SD: {filename}")

        # 2. Encode for RAM cache (for dashboard UI)
        if crop is not None:
            ret, jpeg = cv2.imencode('.jpg', crop)
            if ret:
                shared.image_cache[str(result["track_id"])] = jpeg.tobytes()

        shared.final_logs.insert(0, result)

        # 3. Prevent RAM bloat — keep only last 10 results in memory
        if len(shared.final_logs) > 10:
            oldest = shared.final_logs.pop()
            old_id = str(oldest["track_id"])
            if old_id in shared.image_cache:
                del shared.image_cache[old_id]


def toggle_detection():
    """Toggles the global detection state."""
    if not hasattr(shared, 'is_detecting'):
        shared.is_detecting = False
    shared.is_detecting = not shared.is_detecting
    print(f"Detection state changed to: {shared.is_detecting}")
    return shared.is_detecting


def create_crops_zip():
    """Zips the SAVE_DIR and returns the file path."""
    if not os.path.exists(SAVE_DIR) or not os.listdir(SAVE_DIR):
        return None
    zip_base_path = "/tmp/coral_saved_crops"
    shutil.make_archive(zip_base_path, 'zip', SAVE_DIR)
    return zip_base_path + ".zip"


def load_labels():
    """Populate shared.available_classes entirely from config.DETECTOR_LABELS.

    The detector is the entry point — its classes are what the user controls
    in the GUI checkboxes. The classifier handles its own label mapping
    internally via labels.txt and is downstream of this filter.

    Both shared.allowed_classes and shared.allowed_detector_ids are
    initialised to all detector classes on boot.
    """
    classes = dict(config.DETECTOR_LABELS)  # {0: "car", 1: "person", ...}

    shared.available_classes = classes
    shared.allowed_classes = list(classes.keys())
    shared.allowed_detector_ids = set(classes.keys())

    print(f"Loaded {len(classes)} classes from DETECTOR_LABELS: "
          f"{list(classes.values())}")


def _build_detector_ids_from_selection(allowed_ids):
    """The GUI now uses detector IDs directly, so no mapping needed —
    the selected IDs ARE the detector IDs. Just return them as a set.
    """
    return set(allowed_ids)


def update_allowed_classes(new_list):
    """Update which classes flow through the full pipeline from a single
    user checkbox action in the GUI.

    shared.allowed_classes      — drives the classifier-side bouncer in
                                  results_worker (name-matched, case-insensitive)
    shared.allowed_detector_ids — drives the detector-side filter in
                                  detector.py (ID-matched, zero overhead)

    Since the GUI now uses detector IDs throughout, both lists are the same
    set of integers — no translation needed.
    """
    shared.allowed_classes = new_list
    shared.allowed_detector_ids = _build_detector_ids_from_selection(new_list)

    active_names = [config.DETECTOR_LABELS.get(i, f"id{i}") for i in new_list]
    print(f"Active classes updated: {active_names}")
    return True
