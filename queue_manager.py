import queue

# Queue for sending batches of cropped frames from the Tracker to the Classifier
classifier_batch_queue = queue.Queue(maxsize=50)

# Queue for sending final classified results from the Classifier to the Dashboard/Logger
results_queue = queue.Queue(maxsize=100)
