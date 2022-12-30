import argparse
import time
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import tensorflow as tf

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-m', '--model', required=True, help='File path of .tflite file.')
parser.add_argument(
  '-l', '--labels', help='File path of labels file.')
parser.add_argument(
    '-k', '--top_k', type=int, default=1,
    help='Max number of classification results')
parser.add_argument(
    '-t', '--threshold', type=float, default=0.0,
    help='Classification score threshold')
parser.add_argument(
    '-c', '--count', type=int, default=5,
    help='Number of times to run inference')
parser.add_argument(
    '-a', '--input_mean', type=float, default=128.0,
    help='Mean value for input normalization')
parser.add_argument(
    '-s', '--input_std', type=float, default=128.0,
    help='STD value for input normalization')
args = parser.parse_args()

labels = read_label_file(args.labels) if args.labels else {}

interpreter = make_interpreter(*args.model.split('@'))
interpreter.allocate_tensors()

# Model must be uint8 quantized
if common.input_details(interpreter, 'dtype') != np.uint8:
  raise ValueError('Only support uint8 input type.')

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
image=test_images[50]

params = common.input_details(interpreter, 'quantization_parameters')
scale = params['scales']
zero_point = params['zero_points']
mean = args.input_mean
std = args.input_std
if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
  # Input data does not require preprocessing.
  common.set_input(interpreter, image)
else:
  # Input data requires preprocessing
  normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
  np.clip(normalized_input, 0, 255, out=normalized_input)
  common.set_input(interpreter, normalized_input.astype(np.uint8))

# Run inference
print('----INFERENCE TIME----')
print('Note: The first inference on Edge TPU is slow because it includes',
      'loading the model into Edge TPU memory.')
for _ in range(args.count):
  start = time.perf_counter()
  interpreter.invoke()
  inference_time = time.perf_counter() - start
  classes = classify.get_classes(interpreter, args.top_k, args.threshold)
  print('%.1fms' % (inference_time * 1000))

print('-------RESULTS--------')
for c in classes:
  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))