import config
import cv2
import os
import time
import queue
import numpy as np
import threading
from collections import Counter
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify
from pycoral.utils.dataset import read_label_file
from queue_manager import classifier_batch_queue
import shared

# --- Diagnostic crop saving ---
# Set to True to save best_crop for every finalized vehicle with detector
# and classifier labels in the filename. After a day's running, sort the
# saved folder by filename to see patterns: det_LGV_cls_OGV2 etc.
SAVE_DIAGNOSTIC_CROPS = True
DIAGNOSTIC_DIR = "/mnt/server_output/diagnostic_crops"


def classifier_worker():
    interpreter = make_interpreter(config.MODEL_CLASSIFIER)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    input_details = interpreter.get_input_details()[0]
    scale, zero_point = input_details['quantization']
    print(f"Classifier input quantization -> scale: {scale}, zero_point: {zero_point}")

    labels = read_label_file(config.LABELS_FILE)

    # Detector uses a SEPARATE label ordering from the classifier.
    # Read its own label mapping from config rather than reusing `labels`.
    detector_label_names = config.DETECTOR_LABELS

    if SAVE_DIAGNOSTIC_CROPS:
        os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
        print(f"Diagnostic crops will be saved to: {DIAGNOSTIC_DIR}")

    print("Classifier thread waiting for vehicle batches...")

    while not shared.stop_event.is_set():
        try:
            payload = classifier_batch_queue.get(timeout=0.5)
        except queue.Empty:
            continue  # just gives us a chance to re-check stop_event

        track_id = payload["track_id"]
        crops = payload["crops"]

        # detector_labels is a list of per-frame detector guesses accumulated
        # by the tracker. Majority-vote it the same way we vote classifier predictions.
        raw_detector_labels = payload.get("detector_labels", [])
        if raw_detector_labels:
            detector_label_id, _ = Counter(raw_detector_labels).most_common(1)[0]
            detector_label_name = detector_label_names.get(detector_label_id, f"det{detector_label_id}")
        else:
            detector_label_name = "unknown"

        predictions = []
        scores = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            resized = cv2.resize(crop, (input_width, input_height))
            input_tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            if scale == 1.0 and zero_point == 0:
                quantized_input = input_tensor.astype(np.int8)
            else:
                quantized_input = (
                    input_tensor.astype(np.float32) / scale + zero_point
                ).astype(np.int8)

            common.set_input(interpreter, quantized_input)
            interpreter.invoke()

            classes = classify.get_classes(interpreter, top_k=1)
            if classes:
                predictions.append(classes[0].id)
                scores.append(classes[0].score)

        if predictions:
            most_common_class, votes = Counter(predictions).most_common(1)[0]
            class_name = labels.get(most_common_class, f"Unknown ID {most_common_class}")

            winning_scores = [scores[i] for i, p in enumerate(predictions) if p == most_common_class]
            avg_score = sum(winning_scores) / len(winning_scores) if winning_scores else 0.0

            best_crop = crops[len(crops) // 2]

            print(f"✅ Vehicle {track_id} finalized: {class_name.upper()} "
                  f"(conf: {avg_score:.2f}, det_guess: {detector_label_name})")

            # Save diagnostic crop with both labels in filename
            if SAVE_DIAGNOSTIC_CROPS and best_crop is not None:
                timestamp = int(time.time())
                # Filename pattern: v{id}_det_{detector_guess}_cls_{classifier_guess}_conf{score}.jpg
                # Sort by det_ or cls_ prefix in a file manager to find patterns.
                safe_class = class_name.replace(" ", "_")
                fname = (f"v{track_id:04d}"
                         f"_det_{detector_label_name}"
                         f"_cls_{safe_class}"
                         f"_conf{avg_score:.2f}"
                         f"_{timestamp}.jpg")
                cv2.imwrite(os.path.join(DIAGNOSTIC_DIR, fname), best_crop)

            from queue_manager import results_queue
            results_queue.put({
                "track_id": int(track_id),
                "class_id": int(most_common_class),
                "class_name": class_name.title(),
                "votes": int(votes),
                "frames": int(len(crops)),
                "score": float(avg_score),
                "crop": best_crop,
                "detector_label": detector_label_name,
            })
