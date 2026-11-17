# Assignment 3-2: Object Detection

The program uses `jetson-inference` and SSD MobileNet v2 to detect objects in two images. It was run on a Jetson Nano with TensorRT 8.0.1 at confidence thresholds of 0.3 and 0.7.

With `jetson-inference` and the SSD MobileNet v2 model installed, run:

```bash
python3 assin3_detection.py --threshold 0.3 images/image1.jpg images/image2.jpg
python3 assin3_detection.py --threshold 0.7 images/image1.jpg images/image2.jpg
```

The program saves annotated images and detection records in `results/`. For the person class, the runs detected 4 and 4 people in image 1 and 1 and 1 person in image 2, at thresholds 0.3 and 0.7 respectively. The model can also detect other classes.

The input images are from the [NVIDIA jetson-inference sample images](https://github.com/dusty-nv/jetson-inference/tree/master/data/images).
