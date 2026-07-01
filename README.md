# Coral Traffic Survey V2.0

An embedded, low-power traffic survey system running on a Google Coral Dev Board. This system captures live video, detects and classifies vehicles using quantized Edge TPU models, and monitors its own power draw in real-time.

## 🚀 V2.0 Features
* **Modular Architecture:** Dedicated background threads for camera streaming, detection, classification, and hardware monitoring to maximize Coral CPU/TPU efficiency.
* **Live Web Dashboard:** A Flask-based web GUI featuring real-time diagnostic charts (CPU/RAM usage, TPU temps, and Wattage) using Chart.js.
* **Hardware Power Monitoring:** Integrates directly with a UM25C Bluetooth multimeter to track live inference power draw (averaging ~4.5W under full load).
* **SD Card Logging:** Toggleable UI control to automatically crop and save vehicle detections to the SD card based on confidence thresholds.
* **Hot-Swappable Camera Setup:** Update video paths (`/dev/video1` or RTSP streams) directly from the web interface without restarting the device.

## 💻 Hardware Requirements
* Google Coral Dev Board (NXP i.MX 8M SoC)
* USB UVC Camera
* UM25C Bluetooth Power Meter (Optional, for power monitoring)
* 18650 Battery Bank (For remote deployment)

## 🛠️ Power Monitor Setup (UM25C)
The power monitoring module uses a custom C binary to read data over Bluetooth.
1. Connect the UM25C via Bluetooth: `sudo rfcomm bind rfcomm0 <MAC_ADDRESS>`
2. Compile the binary on the Coral board: `gcc -o um25c um25c.c -lm`
3. Run the main application; `power_monitor.py` will automatically execute the binary and pipe wattage to the web dashboard.

## 🧠 Models
* **Detector:** EfficientDet-Lite0 (Quantized, Edge TPU)
* **Classifier:** MobileNetV3 (Quantized, Edge TPU)
