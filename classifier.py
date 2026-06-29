import cv2
import threading
from collections import Counter
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify
from pycoral.utils.dataset import read_label_file
from queue_manager import classifier_batch_queue

def classifier_worker():
    interpreter = make_interpreter('traffic_classifier_quant_edgetpu.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Read your labels file!
    labels = read_label_file('labels.txt') 

    print("Classifier thread waiting for vehicle batches...")
    
    while True:
        payload = classifier_batch_queue.get()
        track_id = payload["track_id"]
        crops = payload["crops"]
        
        predictions = []
        scores = [] 

        for crop in crops:
            resized = cv2.resize(crop, (input_width, input_height))
            input_tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            common.set_input(interpreter, input_tensor)
            interpreter.invoke()
            
            classes = classify.get_classes(interpreter, top_k=1)
            if classes:
                predictions.append(classes[0].id)
                scores.append(classes[0].score) 

        if predictions:
            most_common_class, votes = Counter(predictions).most_common(1)[0]
            
            # Map the ID to the string name (e.g., 1 -> "car")
            class_name = labels.get(most_common_class, f"Unknown ID {most_common_class}")
            
            winning_scores = [scores[i] for i, p in enumerate(predictions) if p == most_common_class]
            avg_score = sum(winning_scores) / len(winning_scores) if winning_scores else 0.0
            
            best_crop = crops[len(crops) // 2]
            
            print(f"✅ Vehicle {track_id} finalized: {class_name.upper()} "
                  f"(conf: {avg_score:.2f})")
            
            from queue_manager import results_queue
            results_queue.put({
                "track_id": int(track_id),
                "class_id": int(most_common_class),
                "class_name": class_name.title(), # <-- Added human-readable string
                "votes": int(votes),
                "frames": int(len(crops)),
                "score": float(avg_score),
                "crop": best_crop          
            })
