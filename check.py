from pycoral.utils.edgetpu import make_interpreter

interpreter = make_interpreter("traffic_model_edgetpu.tflite")
interpreter.allocate_tensors()
print(interpreter.get_input_details()[0]['quantization'])
