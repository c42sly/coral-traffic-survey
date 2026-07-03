import os
from flask import Flask, jsonify, Response, request, render_template_string, send_file
import controller, shared, queue_manager
import cv2

app = Flask(__name__)

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
        
        /* Settings & Stats */
        .settings-row { display: flex; gap: 10px; margin-top: 10px; margin-bottom: 20px;}
        .settings-row input { flex: 1; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
        
        .stat-badge { background: #e9ecef; padding: 8px 12px; border-radius: 6px; font-size: 14px; border: 1px solid #ced4da; }
        .stat-badge strong { color: #495057; }

        /* Class List */
        .class-list { max-height: 150px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; border-radius: 4px; background: #fff; display: flex; flex-direction: column; gap: 5px; }
        .class-list label { font-size: 14px; cursor: pointer; }
    </style>
</head>
<body>
    <h2>Live Traffic Dashboard (Modular v3.0)</h2>
    
    <div class="panel full-width">
        <div class="controls">
            <h3>System Diagnostics</h3>
            <button id="saveToggle" class="btn btn-green" onclick="toggleSave()">💾 Start Saving to SD Card</button>
        </div>
        
        <div class="settings-row">
            <input type="text" id="cameraUrl" placeholder="Camera Path (e.g. /dev/video1 or rtsp://...)">
            <button class="btn btn-blue" style="margin:0;" onclick="saveCamera()">Update Camera</button>
        </div>
        
        <div class="settings-row">
            <button id="detectToggle" class="btn btn-green" onclick="toggleDetection()">▶️ Start Detection</button>
            <button id="downloadBtn" class="btn btn-blue" onclick="downloadCrops()" disabled style="opacity: 0.5;">📦 Download Crops (ZIP)</button>
        </div>

        <div style="margin-bottom: 20px;">
            <h3>Target Classes</h3>
            <div id="classCheckboxes" class="class-list">
                Loading classes...
            </div>
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
            <div class="controls">
                <h3>Latest Classifications</h3>
                <div style="display: flex; gap: 10px;">
                    <div class="stat-badge"><strong>Session Count:</strong> <span id="sessionCountEl">0</span></div>
                    <div class="stat-badge"><strong>Current Rate:</strong> <span id="sessionRateEl">0 veh/hr</span></div>
                </div>
            </div>
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
        
        // --- Start/Stop Detection Logic ---
        let isDetecting = false;

        function updateDetectUI(detecting) {
            isDetecting = detecting;
            const btn = document.getElementById('detectToggle');
            const dlBtn = document.getElementById('downloadBtn');
            
            if (detecting) {
                btn.innerText = "⏸️ Pause Detection";
                btn.className = "btn btn-red";
                dlBtn.disabled = true;
                dlBtn.style.opacity = "0.5";
                dlBtn.title = "Pause detection to download crops.";
            } else {
                btn.innerText = "▶️ Start Detection";
                btn.className = "btn btn-green";
                dlBtn.disabled = false;
                dlBtn.style.opacity = "1";
                dlBtn.title = "";
            }
        }

        function toggleDetection() {
            fetch('/api/toggle_detection', { method: 'POST' })
                .then(r => r.json())
                .then(data => updateDetectUI(data.detecting));
        }

        function downloadCrops() {
            if (isDetecting) {
                alert("Please pause detection first to protect system stability.");
                return;
            }
            window.location.href = '/api/download_crops';
        }

        // --- Class Selection Logic ---
        function loadClassCheckboxes() {
            fetch('/api/config/classes')
            .then(r => {
                if (!r.ok) throw new Error("API not ready yet");
                return r.json();
            })
            .then(data => {
                const container = document.getElementById('classCheckboxes');
                container.innerHTML = ''; 
                
                for (const [id, name] of Object.entries(data.available)) {
                    const isChecked = data.allowed.includes(parseInt(id)) ? 'checked' : '';
                    container.innerHTML += `
                        <label>
                            <input type="checkbox" value="${id}" onchange="saveClasses()" ${isChecked}> 
                            ${name.toUpperCase()} (ID: ${id})
                        </label>
                    `;
                }
            })
            .catch(err => {
                console.log("Waiting for backend class list...");
            });
        }

        function saveClasses() {
            const checkboxes = document.querySelectorAll('#classCheckboxes input[type="checkbox"]:checked');
            const allowedIds = Array.from(checkboxes).map(cb => parseInt(cb.value));
            
            fetch('/api/config/classes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ allowed: allowedIds })
            });
        }

        // --- Session Counting Logic ---
        let sessionStartTime = new Date();
        let seenTrackIds = new Set();
        let sessionCount = 0;

        // --- Results Logic ---
        function updateResults() {
            fetch('/api/results').then(r => r.json()).then(data => {
                const list = document.getElementById('results-list');
                if(data.length === 0) return;
                
                // Track unique vehicles and calculate count
                data.forEach(d => {
                    if (!seenTrackIds.has(d.track_id)) {
                        seenTrackIds.add(d.track_id);
                        sessionCount++;
                    }
                });

                // Calculate Elapsed Time
                let now = new Date();
                let elapsedSeconds = (now - sessionStartTime) / 1000;
                let elapsedHours = elapsedSeconds / 3600;
                
                // Update the total count immediately
                document.getElementById('sessionCountEl').innerText = sessionCount;

                // Wait 30 seconds to gather a baseline before showing the hourly rate
                if (elapsedSeconds < 30) {
                    document.getElementById('sessionRateEl').innerText = "Gathering baseline...";
                } else {
                    let rate = Math.round(sessionCount / elapsedHours);
                    document.getElementById('sessionRateEl').innerText = rate + " veh/hr";
                }
                
                list.innerHTML = data.map(d => `
                    <div class="result-card">
                        <img src="/crop/${d.track_id}" class="result-thumb" onerror="this.style.display='none'">
                        <div class="result-info">
                            <strong>🚗 Vehicle ID: ${d.track_id} | Class ID: ${d.class_name}</strong>
                            <span style="color: #666; font-size: 14px;">
                                Confidence: ${(d.score * 100).toFixed(1)}% | ⏱️ Time: ${d.timestamp || 'N/A'}
                            </span>
                        </div>
                    </div>
                `).join('');
            });
        }

        // On Load initialization
        fetch('/api/save_status').then(r => r.json()).then(data => updateSaveBtnUI(data.saving));
        fetch('/api/detect_status').then(r => r.json()).then(data => updateDetectUI(data.detecting)); 
        loadCameraSettings();
        loadClassCheckboxes();

        setInterval(updateStats, 2000); 
        setInterval(updateResults, 1000); 
    </script>
</body>
</html>
"""

@app.route('/api/stats')
def api_stats():
    stats = controller.get_sys_stats()
    return jsonify(stats)

@app.route('/api/toggle_save', methods=['POST'])
def api_toggle_save():
    status = controller.toggle_save()
    return jsonify({"saving": status})

@app.route('/api/config/camera', methods=['GET','POST'])
def api_camera_config():
    if request.method == 'POST':
        new_url = request.json.get('url')
        if new_url:
            success = controller.set_camera_url(new_url)
            return jsonify({"ok": success})
        return jsonify({"error": "No URL provided"}), 400
    else:
        # GET current camera
        return jsonify({"url": controller.get_config_camera()})

@app.route('/snapshot')
def snapshot():
    with shared.lock:
        frame = shared.latest_frame
        if frame is None:
            return "Initializing...", 503
        frame_copy = frame.copy()
    ret, jpeg = cv2.imencode('.jpg', frame_copy)
    return Response(jpeg.tobytes(), mimetype='image/jpeg') if ret else ("Error encoding", 500)

@app.route('/crop/<track_id>')
def serve_crop(track_id):
    # Returns the cached image bytes from shared state
    image_bytes = shared.image_cache.get(track_id)
    if image_bytes:
        return Response(image_bytes, mimetype='image/jpeg')
    return "Not found", 404

@app.route('/api/results')
def api_results():
    # Returns the latest classification logs directly from shared state
    return jsonify(shared.final_logs)

@app.route('/api/toggle_detection', methods=['POST'])
def api_toggle_detection():
    status = controller.toggle_detection()
    return jsonify({"detecting": status})

@app.route('/api/detect_status')
def api_detect_status():
    status = getattr(shared, 'is_detecting', False)
    return jsonify({"detecting": status})

@app.route('/api/download_crops')
def api_download_crops():
    if getattr(shared, 'is_detecting', False):
        return "System must be paused to download.", 403
    
    zip_path = controller.create_crops_zip()
    if zip_path and os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True, download_name="coral_crops.zip")
    return "No crops found.", 404

@app.route('/api/config/classes', methods=['GET', 'POST'])
def api_classes():
    if request.method == 'POST':
        new_classes = request.json.get('allowed', [])
        new_classes = [int(x) for x in new_classes]
        # This calls the controller function we need to add!
        controller.update_allowed_classes(new_classes)
        return jsonify({"ok": True})
    else:
        return jsonify({
            "available": getattr(shared, 'available_classes', {}),
            "allowed": getattr(shared, 'allowed_classes', [])
        })

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)
