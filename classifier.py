import config
import cv2
import numpy as np
import threading
from collections import Counter
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify
from pycoral.utils.dataset import read_label_file
from queue_manager import classifier_batch_queue


def classifier_worker():
    interpreter = make_interpreter(config.MODEL_CLASSIFIER)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Read the model's actual quantization params instead of hardcoding them,
    # so this keeps working if the model is ever recalibrated/retrained.
    input_details = interpreter.get_input_details()[0]
    scale, zero_point = input_details['quantization']
    print(f"Classifier input quantization -> scale: {scale}, zero_point: {zero_point}")

    labels = read_label_file(config.LABELS_FILE)
    print("Classifier thread waiting for vehicle batches...")

    while True:
        payload = classifier_batch_queue.get()
        track_id = payload["track_id"]
        crops = payload["crops"]

        predictions = []
        scores = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            resized = cv2.resize(crop, (input_width, input_height))
            input_tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # input_tensor is uint8 in [0, 255] (real pixel values).
            # The model's int8 input tensor expects: quantized = real/scale + zero_point.
            # Without this step, set_input() just reinterprets the raw uint8 bytes
            # as int8, which silently corrupts every pixel value.
            if scale == 1.0 and zero_point == 0:
                # Model genuinely expects raw uint8 reinterpreted as int8 (rare, but
                # cheap to special-case and skip the float math when it's true).
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

            print(f"✅ Vehicle {track_id} finalized: {class_name.upper()} (conf: {avg_score:.2f})")

            from queue_manager import results_queue
            results_queue.put({
                "track_id": int(track_id),
                "class_id": int(most_common_class),
                "class_name": class_name.title(),
                "votes": int(votes),
                "frames": int(len(crops)),
                "score": float(avg_score),
                "crop": best_crop
            })
