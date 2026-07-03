# Coral Traffic Survey V3.0 (Modular Refactor)

An embedded, low-power traffic survey system built around the Google Coral Dev Board. V3.0 features a completely overhauled modular architecture that decouples core computer vision logic from the web interface for improved fault tolerance and maintainability.

## 🏗️ V3.0 Architecture & Module Roles
The system utilizes a controller-GUI design pattern to safely handle concurrent, multi-threaded pipelines on embedded hardware:

* **`app.py`**: A minimal application launcher responsible solely for executing the controller thread setup and starting the Flask development server.
* **`controller.py`**: The central orchestrator of the system. It spawns and manages the lifecycles of background threads (detector, classifier, and results logs), safely aggregates system diagnostics, and mutates shared system configurations.
* **`web_gui.py`**: A dedicated presentation layer containing all Flask routes and REST API endpoints. It queries shared system states or invokes controller interfaces without directly interfering with thread execution or processing queues.
* **`shared.py`**: Holds thread-safe global flags, locks, and RAM caches (such as live frames and recent vehicle crops) accessed concurrently by the UI and processing pipelines.

* ## 🚦 Usage & Lifecycle

### Startup
To start the entire application (including backend ML pipelines and the web server), execute the launcher script:
```bash
python3 app.py

🧪 Verifying Endpoints (Testing)
If the web browser dashboard fails to render, verify that your backend APIs are actively emitting data from the Coral board using `curl`:

```bash
# Verify system stats data stream
curl http://localhost:5000/api/stats

# Verify backend classification queue array
curl http://localhost:5000/api/results

## 📝 CHANGELOG
* **V3.0 (Current Refactor):** Complete decoupled refactor. Separated application controller orchestration logic from Flask presentation layer to safeguard multi-threaded performance. Implemented thread-safe memory logging caches and optimized absolute path I/O routing directly to external SD card storage mounts to conserve system flash storage memory.
* **V2.01 (Previous Stable):** Monolithic single-file runtime pipeline featuring dual Edge TPU inference mapping, tracking, live dashboard integration, and raw diagnostic output.
