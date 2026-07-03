import threading
import shared, queue_manager, detector, classifier, time
import re
import config
import os
import cv2
import shutil
import os
from flask import send_file # You'll need this passed back to the GUI

_threads = []

# Ensure our save directory exists
SAVE_DIR = getattr(config, 'SAVE_DIR', 'saved_crops')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

def start():
    """Start detector, classifier, results, (power) threads, and load labels."""
    # --- NEW: Actually load the labels into RAM before starting ---
    load_labels()
    
    # Detector thread
    det_thread = threading.Thread(target=detector.inference_loop, daemon=True)
    det_thread.start()
    _threads.append(det_thread)
    
    # Classifier thread
    class_thread = threading.Thread(target=classifier.classifier_worker, daemon=True)
    class_thread.start()
    _threads.append(class_thread)
    
    # Results worker thread 
    res_thread = threading.Thread(target=results_worker, daemon=True)
    res_thread.start()
    _threads.append(res_thread)
    
    print("All threads started.")

def stop():
    """Signal threads to stop and wait for them."""
    shared.running = False  # Suppose loops check this flag
    for t in _threads:
        t.join(timeout=1.0)
    print("Threads stopped.")

def toggle_save():
    shared.save_to_sd = not shared.save_to_sd
    return shared.save_to_sd

def set_camera_url(url):
    try:
        with open('config.py','r') as f: 
            content=f.read()
        content_new = re.sub(r"^VIDEO_DEVICE = .*$", f"VIDEO_DEVICE = '{url}'", content, flags=re.MULTILINE)
        with open('config.py','w') as f: 
            f.write(content_new)
    except Exception as e:
        return False
    shared.requested_camera_url = url
    return True

def get_config_camera():
    try:
        return config.VIDEO_DEVICE
    except AttributeError:
        return ""

def get_sys_stats():
    stats = {"cpu_usage": 0, "cpu_temp": 0, "tpu_temp": 0, "ram_usage": 0, "power_watts": 0.0}
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
            stats["ram_usage"] = round(((mem_total - mem_available) / mem_total) * 100, 1)
    except: pass
    
    # Grab the power reading safely using the lock
    with shared.lock:
        stats["power_watts"] = getattr(shared, 'current_power_watts', 0.0)
         
    return stats

# --- Background Worker to Process Results ---
def results_worker():
    print("Background result writer started.")
    
    while True:
        result = queue_manager.results_queue.get()
        crop = result.pop("crop", None) 
        
        # Inject the current time into the result
        result['timestamp'] = time.strftime("%H:%M:%S")
        
        # --- THE BOUNCER (Now Case-Insensitive) ---
        if hasattr(shared, 'allowed_classes') and hasattr(shared, 'available_classes'):
            # Convert our allowed list to lowercase
            allowed_names = [str(shared.available_classes.get(cid)).lower() for cid in shared.allowed_classes]
            
            # Convert the AI's guess to lowercase
            ai_guess = str(result.get('class_name', '')).lower()
            
            # If the guess isn't in the list, drop it!
            if ai_guess not in allowed_names:
                continue 

        # 1. Save to SD Card (If toggled ON)
        if shared.save_to_sd and crop is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_class = result['class_name'].replace(' ', '')
            conf = int(result['score'] * 100)
            
            filename = f"{timestamp}_vid{result['track_id']}_{safe_class}_conf{conf}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            
            cv2.imwrite(filepath, crop)
            print(f"💾 Saved crop to SD: {filename}")

        # 2. Encode for RAM cache (For Dashboard UI)
        if crop is not None:
            ret, jpeg = cv2.imencode('.jpg', crop)
            if ret:
                shared.image_cache[str(result["track_id"])] = jpeg.tobytes()
                
        shared.final_logs.insert(0, result)
        
        # 3. Prevent RAM bloat by only keeping the last 10 in memory
        if len(shared.final_logs) > 10:
            oldest = shared.final_logs.pop()
            old_id = str(oldest["track_id"])
            if old_id in shared.image_cache:
                del shared.image_cache[old_id]

def toggle_detection():
    """Toggles the global detection state."""
    if not hasattr(shared, 'is_detecting'):
        shared.is_detecting = False # Default to paused on cold boot
        
    shared.is_detecting = not shared.is_detecting
    print(f"Detection state changed to: {shared.is_detecting}")
    return shared.is_detecting

def create_crops_zip():
    """Zips the SAVE_DIR and returns the file path."""
    if not os.path.exists(SAVE_DIR) or not os.listdir(SAVE_DIR):
        return None
    
    # We save the zip to /tmp so it builds in RAM and deletes on reboot
    zip_base_path = "/tmp/coral_saved_crops"
    shutil.make_archive(zip_base_path, 'zip', SAVE_DIR)
    
    return zip_base_path + ".zip"

def load_labels():
    """Reads labels.txt and populates the shared available_classes dict."""
    classes = {}
    try:
        with open(config.LABELS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # Split "0 car" into key 0, value "car"
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    # Strip out any weird formatting if it exists
                    clean_name = parts[1].replace('', '').strip()
                    classes[int(parts[0])] = clean_name
                    
        shared.available_classes = classes
        # By default on boot, allow ALL classes
        shared.allowed_classes = list(classes.keys())
        print(f"Loaded {len(classes)} classes from {config.LABELS_FILE}")
    except Exception as e:
        print(f"Error loading labels: {e}")

def update_allowed_classes(new_list):
    """Updates the allowed classes list from the web GUI."""
    shared.allowed_classes = new_list
    print(f"Tracking classes updated: {shared.allowed_classes}")
    return True
