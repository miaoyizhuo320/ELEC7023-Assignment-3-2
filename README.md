# Assignment 3-2: Object Detection

`assin3_detection.py` uses the `jetson-inference` SSD MobileNet v2 detector on Jetson. It has not been tested on a Jetson device. Run it in an environment where `jetson-inference` and its models are installed:

```bash
python3 assin3_detection.py --threshold 0.3 images/image1.jpg images/image2.jpg
python3 assin3_detection.py --threshold 0.7 images/image1.jpg images/image2.jpg
```

`simulate_mac.py` produced the report results on a Mac with TensorFlow CPU, using TensorFlow's `ssd_mobilenet_v2_coco_2018_03_29` model. Its weights and runtime are not verified to be identical to the Jetson model. The frozen graph applies a 0.3 score cutoff and non-maximum suppression; the 0.7 output is filtered from the same graph output. The report counts the `person` class only.

To reproduce the Mac results using Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_mac.txt
python simulate_mac.py
```

The script expects `ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb` in this directory. The full local package includes it. If it is not included in an online copy, download the model archive from the [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and extract it here. Test images are from [NVIDIA jetson-inference sample images](https://github.com/dusty-nv/jetson-inference/tree/master/data/images).
