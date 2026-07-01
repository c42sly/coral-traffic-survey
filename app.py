import threading
import time
import os
import cv2
import re
from flask import Flask, jsonify, Response, render_template_string, request

# --- IMPORT YOUR CUSTOM MODULES ---
import detector
import classifier
from queue_manager import results_queue
import shared
import power_monitor

app = Flask(__name__)
final_logs = []
image_cache = {} # RAM Cache for the UI vehicle crops

# --- NEW: SD Card Saving Config ---
SAVE_DIR = os.path.expanduser("~/mnt/server_output")
os.makedirs(SAVE_DIR, exist_ok=True)
save_to_sd = False # Global toggle state

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
    global save_to_sd
    print("Background result writer started.")
    
    while True:
        result = results_queue.get()
        crop = result.pop("crop", None) 
        
        # 1. Save to SD Card (If toggled ON)
        if save_to_sd and crop is not None:
            # Format: YYYYMMDD_HHMMSS_vid12_classCar_conf85.jpg
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
                image_cache[str(result["track_id"])] = jpeg.tobytes()
                
        final_logs.insert(0, result)
        
        # 3. Prevent RAM bloat by only keeping the last 10 in memory
        if len(final_logs) > 10:
            oldest = final_logs.pop()
            old_id = str(oldest["track_id"])
            if old_id in image_cache:
                del image_cache[old_id]

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Coral Traffic Dashboard V2</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f9; padding: 20px; }
        .grid { display: flex; flex-wrap: wrap; gap: 20px; }
        .panel { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 300px; }
        .full-width { width: 100%; margin-bottom: 20px; }
        
        /* Buttons */
        .btn { padding: 10px 15px; color: white; border: none; border-radius: 4px; cursor: pointer; margin-bottom: 10px; font-weight: bold;}
        .btn-blue { background: #007bff; }
        .btn-green { background: #28a745; }
        .btn-red { background: #dc3545; }
        
        .snapshot-img { max-width: 100%; border: 1px solid #ccc; background: #eee; min-height: 200px;}
        .controls { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;}
        
        #results-list { display: flex; flex-direction: column; gap: 10px; }
        .result-card { display: flex; align-items: center; gap: 15px; padding: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }
        .result-thumb { width: 80px; height: 80px; object-fit: cover; border-radius: 4px; border: 1px solid #ccc; background: #eee; }
        .result-info { display: flex; flex-direction: column; gap: 5px;}
        
        /* NEW: Settings inputs */
        .settings-row { display: flex; gap: 10px; margin-top: 10px; margin-bottom: 20px;}
        .settings-row input { flex: 1; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
    </style>
</head>
<body>
    <h2>Live Traffic Dashboard (Modular V2)</h2>
    
    <div class="panel full-width">
        <div class="controls">
            <h3>System Diagnostics</h3>
            <button id="saveToggle" class="btn btn-green" onclick="toggleSave()">💾 Start Saving to SD Card</button>
        </div>
        
        <div class="settings-row">
            <input type="text" id="cameraUrl" placeholder="Camera Path (e.g. /dev/video1 or rtsp://...)">
            <button class="btn btn-blue" style="margin:0;" onclick="saveCamera()">Update Camera</button>
        </div>
        
        <canvas id="sysChart" height="60"></canvas>
    </div>

    <div class="grid">
        <div class="panel">
            <div class="controls">
                <h3>Live Detector Framing</h3>
                <label>
                    <input type="checkbox" id="autoRefresh" onchange="toggleAutoRefresh()"> Auto-Refresh Feed
                </label>
            </div>
            <button class="btn btn-blue" onclick="updateSnapshot()">Grab Single Frame</button><br>
            <img id="snapshot" class="snapshot-img" src="/snapshot" onerror="this.style.display='none'" onload="this.style.display='block'">
        </div>

        <div class="panel" style="flex: 2;">
            <h3>Latest Classifications</h3>
            <div id="results-list">Waiting for finalized vehicles...</div>
        </div>
    </div>

    <script>
        // --- Graph Logic ---
        const ctx = document.getElementById('sysChart').getContext('2d');
        const sysChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'CPU Usage (%)', borderColor: '#007bff', data: [], fill: false, tension: 0.1 },
                    { label: 'RAM Usage (%)', borderColor: '#fd7e14', data: [], fill: false, tension: 0.1 },
                    { label: 'CPU Temp (°C)', borderColor: '#dc3545', data: [], fill: false, tension: 0.1 },
                    { label: 'TPU Temp (°C)', borderColor: '#28a745', data: [], fill: false, tension: 0.1 },
                    { label: 'Power Draw (W)', borderColor: '#6f42c1', data: [], fill: false, tension: 0.1 }
                ]
            },
            options: { animation: false, scales: { y: { suggestedMin: 0, suggestedMax: 100 } } }
        });

        function updateStats() {
            fetch('/api/stats').then(r => r.json()).then(data => {
                const now = new Date().toLocaleTimeString();
                if (sysChart.data.labels.length > 20) {
                    sysChart.data.labels.shift();
                    sysChart.data.datasets.forEach(d => d.data.shift());
                }
                sysChart.data.labels.push(now);
                sysChart.data.datasets[0].data.push(data.cpu_usage);
                sysChart.data.datasets[1].data.push(data.ram_usage);
                sysChart.data.datasets[2].data.push(data.cpu_temp);
                sysChart.data.datasets[3].data.push(data.tpu_temp);
                sysChart.data.datasets[4].data.push(data.power_watts);
                sysChart.update();
            });
        }

        // --- Live Feed Logic ---
        let snapshotInterval = null;
        
        function updateSnapshot() {
            const img = document.getElementById('snapshot');
            img.src = '/snapshot?' + new Date().getTime(); 
        }

        function toggleAutoRefresh() {
            const checkbox = document.getElementById('autoRefresh');
            if (checkbox.checked) {
                snapshotInterval = setInterval(updateSnapshot, 500); 
            } else {
                clearInterval(snapshotInterval);
            }
        }

        // --- SD Card Toggle Logic ---
        function updateSaveBtnUI(isSaving) {
            const btn = document.getElementById('saveToggle');
            if (isSaving) {
                btn.innerText = "🛑 Stop Saving to SD Card";
                btn.className = "btn btn-red";
            } else {
                btn.innerText = "💾 Start Saving to SD Card";
                btn.className = "btn btn-green";
            }
        }

        function toggleSave() {
            fetch('/api/toggle_save', { method: 'POST' })
                .then(r => r.json())
                .then(data => updateSaveBtnUI(data.saving));
        }

        // --- Camera Config Logic ---
        function loadCameraSettings() {
            fetch('/api/config/camera').then(r => r.json()).then(data => {
                document.getElementById('cameraUrl').value = data.url;
            });
        }

        function saveCamera() {
            const newUrl = document.getElementById('cameraUrl').value;
            fetch('/api/config/camera', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: newUrl })
            }).then(r => r.json()).then(data => {
                alert("Camera updated to: " + data.url);
            });
        }

        // --- Results Logic ---
        function updateResults() {
            fetch('/api/results').then(r => r.json()).then(data => {
                const list = document.getElementById('results-list');
                if(data.length === 0) return;
                
                list.innerHTML = data.map(d => `
                    <div class="result-card">
                        <img src="/crop/${d.track_id}" class="result-thumb" onerror="this.style.display='none'">
                        <div class="result-info">
                            <strong>🚗 Vehicle ID: ${d.track_id} | Class ID: ${d.class_name}</strong>
                            <span style="color: #666; font-size: 14px;">Confidence: ${(d.score * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                `).join('');
            });
        }

        // On Load initialization
        fetch('/api/save_status').then(r => r.json()).then(data => updateSaveBtnUI(data.saving));
        loadCameraSettings();

        setInterval(updateStats, 2000); 
        setInterval(updateResults, 1000); 
    </script>
</body>
</html>
"""

# --- Routes ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stats')
def api_stats():
    return jsonify(get_sys_stats())

@app.route('/api/toggle_save', methods=['POST'])
def api_toggle_save():
    global save_to_sd
    save_to_sd = not save_to_sd
    return jsonify({"saving": save_to_sd})

@app.route('/api/save_status')
def api_save_status():
    return jsonify({"saving": save_to_sd})

@app.route('/api/config/camera', methods=['GET', 'POST'])
def api_camera_config():
    if request.method == 'POST':
        new_url = request.json.get('url')
        if new_url:
            # 1. Update config.py so it survives reboots
            try:
                with open('config.py', 'r') as f:
                    content = f.read()
                content = re.sub(r"VIDEO_DEVICE\s*=\s*['\"].*?['\"]", f"VIDEO_DEVICE = '{new_url}'", content)
                with open('config.py', 'w') as f:
                    f.write(content)
            except Exception as e:
                print("Could not write config.py:", e)

            # 2. Tell the detector thread to hot-swap the camera
            with shared.lock:
                shared.requested_camera_url = new_url

            return jsonify({"status": "success", "url": new_url})
    
    # GET request (populate the UI input box)
    try:
        with open('config.py', 'r') as f:
            content = f.read()
        match = re.search(r"VIDEO_DEVICE\s*=\s*['\"](.*?)['\"]", content)
        current_url = match.group(1) if match else ""
    except:
        current_url = ""
    return jsonify({"url": current_url})

@app.route('/api/results')
def api_results():
    return jsonify(final_logs)

@app.route('/crop/<track_id>')
def serve_crop(track_id):
    if track_id in image_cache:
        return Response(image_cache[track_id], mimetype='image/jpeg')
    return "Not found", 404

@app.route('/snapshot')
def snapshot():
    with shared.lock:
        if shared.latest_frame is None: 
            return "Initializing...", 503
        frame_copy = shared.latest_frame.copy()
        
    ret, jpeg = cv2.imencode('.jpg', frame_copy)
    return Response(jpeg.tobytes(), mimetype='image/jpeg') if ret else ("Error", 500)

if __name__ == '__main__':
    print("Starting system threads...")
    
    # Start Power Monitor Thread
    power_thread = threading.Thread(target=power_monitor.power_worker, daemon=True) 
    power_thread.start() 

    # Start the Results writer thread
    writer_thread = threading.Thread(target=results_worker, daemon=True)
    writer_thread.start()

    # Start the ML threads
    class_thread = threading.Thread(target=classifier.classifier_worker, daemon=True)
    class_thread.start()

    det_thread = threading.Thread(target=detector.inference_loop, daemon=True)
    det_thread.start()

    print("Starting Web Server. Open http://<CORAL_IP>:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, threaded=True)
