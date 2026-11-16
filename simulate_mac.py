import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Run SSD MobileNet V2 on two images using TensorFlow.')
parser.add_argument('--model', type=Path, default=base / 'ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb')
parser.add_argument('--labels', type=Path, default=base / 'coco_labels.txt')
parser.add_argument('--image1', type=Path, default=base / 'images/image1.jpg')
parser.add_argument('--image2', type=Path, default=base / 'images/image2.jpg')
parser.add_argument('--output', type=Path, default=base / 'results')
args = parser.parse_args()
labels = args.labels.read_text().splitlines()
args.output.mkdir(parents=True, exist_ok=True)
graph = tf.Graph()
with graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(args.model.read_bytes())
    tf.import_graph_def(graph_def, name='')
tensors = [graph.get_tensor_by_name(name + ':0') for name in
           ('num_detections', 'detection_boxes', 'detection_scores', 'detection_classes')]
input_tensor = graph.get_tensor_by_name('image_tensor:0')
try:
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 28)
except OSError:
    font = ImageFont.load_default()
report = {
    'method': 'TensorFlow CPU frozen graph on Mac, not Jetson/TensorRT',
    'model': 'ssd_mobilenet_v2_coco_2018_03_29',
    'model_source': 'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md',
    'image_source': 'https://github.com/dusty-nv/jetson-inference/tree/master/data/images',
    'results': []
}
with tf.compat.v1.Session(graph=graph) as session:
    for image_idx, path in enumerate((args.image1, args.image2), start=1):
        image = Image.open(path).convert('RGB')
        width, height = image.size
        n, boxes, scores, classes = session.run(tensors, feed_dict={input_tensor: np.asarray(image)[None]})
        candidates = []
        for box, score, cls in zip(boxes[0][:int(n[0])], scores[0][:int(n[0])], classes[0][:int(n[0])]):
            if score < 0.3:
                continue
            top, left, bottom, right = map(float, box)
            left, right, top, bottom = left * width, right * width, top * height, bottom * height
            class_id = int(cls)
            candidates.append({
                'ClassID': class_id, 'Class': labels[class_id] if class_id < len(labels) else 'unknown',
                'Confidence': float(score), 'Left': left, 'Top': top, 'Right': right, 'Bottom': bottom,
                'Width': right - left, 'Height': bottom - top, 'Area': (right - left) * (bottom - top),
                'Center': [(left + right) / 2, (top + bottom) / 2]
            })
        for threshold in (0.3, 0.7):
            detected = [d for d in candidates if d['Confidence'] >= threshold]
            persons = [d for d in detected if d['ClassID'] == 1]
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            for d in persons:
                x0, y0, x1, y1 = [int(round(d[k])) for k in ('Left', 'Top', 'Right', 'Bottom')]
                draw.rectangle((x0, y0, x1, y1), outline='white', width=8)
                draw.rectangle((x0, y0, x1, y1), outline='black', width=4)
                label = 'person %.1f%%' % (100 * d['Confidence'])
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                label_x = max(0, min(x0, width - text_width - 4))
                label_y = max(0, min(y0 - text_height - 4, height - text_height - 4))
                draw.rectangle((label_x, label_y, label_x + text_width + 3, label_y + text_height + 3), fill='white')
                draw.text((label_x + 2 - bbox[0], label_y + 2 - bbox[1]), label, fill='black', font=font)
            filename = 'image%d_threshold_%s.png' % (image_idx, str(threshold).replace('.', 'p'))
            canvas.save(args.output / filename)
            report['results'].append({
                'image': 'image%d' % image_idx, 'image_size': [width, height], 'threshold': threshold,
                'annotated_image': filename, 'person_count': len(persons),
                'person_detections': persons, 'all_detections': detected
            })
            print('image%d threshold %.1f: %d persons, %d detections' %
                  (image_idx, threshold, len(persons), len(detected)), flush=True)
(args.output / 'detections.json').write_text(json.dumps(report, indent=2) + '\n')
