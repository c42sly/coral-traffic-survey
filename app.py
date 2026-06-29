import threading
import time
import os
import cv2
from flask import Flask, jsonify, Response, render_template_string

# --- IMPORT YOUR CUSTOM MODULES ---
import detector
import classifier
from queue_manager import results_queue
import shared

app = Flask(__name__)
final_logs = []
image_cache = {} # <-- NEW: RAM Cache for the vehicle crops

def get_sys_stats():
    stats = {"cpu_usage": 0, "cpu_temp": 0, "tpu_temp": 0, "ram_usage": 0}
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
    return stats

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
        button { padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-bottom: 10px;}
        .snapshot-img { max-width: 100%; border: 1px solid #ccc; }
        
        /* NEW: Styles for the crop thumbnails and lists */
        #results-list { display: flex; flex-direction: column; gap: 10px; }
        .result-card { display: flex; align-items: center; gap: 15px; padding: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }
        .result-thumb { width: 80px; height: 80px; object-fit: cover; border-radius: 4px; border: 1px solid #ccc; background: #eee; }
        .result-info { display: flex; flex-direction: column; gap: 5px;}
    </style>
</head>
<body>
    <h2>Live Traffic Dashboard (Modular V2)</h2>
    
    <div class="panel full-width">
        <h3>System Diagnostics</h3>
        <canvas id="sysChart" height="50"></canvas>
    </div>

    <div class="grid">
        <div class="panel">
            <h3>Camera Framing Snapshot</h3>
            <button onclick="updateSnapshot()">Grab Latest Frame</button><br>
            <img id="snapshot" class="snapshot-img" src="" style="display:none;">
        </div>

        <div class="panel" style="flex: 2;">
            <h3>Latest Classifications</h3>
            <div id="results-list">Waiting for finalized vehicles...</div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('sysChart').getContext('2d');
        const sysChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'CPU Usage (%)', borderColor: '#007bff', data: [], fill: false, tension: 0.1 },
                    { label: 'RAM Usage (%)', borderColor: '#fd7e14', data: [], fill: false, tension: 0.1 },
                    { label: 'CPU Temp (°C)', borderColor: '#dc3545', data: [], fill: false, tension: 0.1 },
                    { label: 'TPU Temp (°C)', borderColor: '#28a745', data: [], fill: false, tension: 0.1 }
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
                sysChart.update();
            });
        }

        function updateSnapshot() {
            const img = document.getElementById('snapshot');
            img.src = '/snapshot?' + new Date().getTime();
            img.style.display = 'block';
        }

        // --- NEW: Display the image and confidence score ---
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

@app.route('/api/results')
def api_results():
    while not results_queue.empty():
        result = results_queue.get()
        
        # 1. Pull the raw numpy crop out of the dictionary (so jsonify doesn't crash)
        crop = result.pop("crop", None) 
        
        # 2. Encode it to a JPEG and store it in the RAM cache
        if crop is not None:
            ret, jpeg = cv2.imencode('.jpg', crop)
            if ret:
                image_cache[str(result["track_id"])] = jpeg.tobytes()
                
        final_logs.insert(0, result)
        
    # Keep only the last 10 records and clear old images from RAM
    if len(final_logs) > 10:
        oldest = final_logs.pop()
        old_id = str(oldest["track_id"])
        if old_id in image_cache:
            del image_cache[old_id]
            
    return jsonify(final_logs)

# --- NEW ROUTE: Serve the vehicle crops to the dashboard ---
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

    class_thread = threading.Thread(target=classifier.classifier_worker, daemon=True)
    class_thread.start()

    det_thread = threading.Thread(target=detector.inference_loop, daemon=True)
    det_thread.start()

    print("Starting Web Server. Open http://<CORAL_IP>:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, threaded=True)
