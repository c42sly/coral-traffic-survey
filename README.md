This project is an embedded traffic survey system running on a Google Coral Dev Board.



Current hardware:

- Google Coral Dev Board

- USB UVC camera (currently /dev/video1)



Current detector:

- EfficientDet-Lite0

- Fully quantized TensorFlow Lite

- Compiled for Edge TPU

- traffic_model_edgetpu.tflite



Current classifier:

- MobileNetV3 classifier

- Fully quantized

- Compiled for Edge TPU

- traffic_classifier_quant_edgetpu.tflite



Current software:

- Python

- Flask dashboard

- OpenCV

- PyCoral

- Live MJPEG camera stream

- Detection snapshots

- System diagnostics



Goal:



Create a modular application with:



camera.py

detector.py

classifier.py

tracker.py

dashboard.py

shared.py

config.py



Architecture:



Camera

↓



Detector



↓



Crop Queue



↓



Classifier



↓



Dashboard



Future versions will include:



Centroid tracker

Vehicle counting

CSV export

Statistics

Multiple cameras



The priority is clean modular code that runs efficiently on the Coral Dev Board. 

traffic_v2/



    app.py              <-- starts everything



    detector.py         <-- Coral detector thread



    classifier.py       <-- Coral classifier thread



    tracker.py          <-- centroid tracker (later)



    dashboard.py        <-- Flask



    models.py           <-- loads TPU interpreters



    camera.py           <-- your VideoStream class



    preprocess.py       <-- letterbox/stretch code



    labels.py           <-- label helpers



    config.py           <-- model names etc.



    queue_manager.py    <-- queues/shared objects
