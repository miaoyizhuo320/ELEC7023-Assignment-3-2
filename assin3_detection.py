import argparse
from pathlib import Path

try:
    from jetson.inference import detectNet
    from jetson.utils import loadImage, saveImage, cudaDeviceSynchronize
except ImportError:
    from jetson_inference import detectNet
    from jetson_utils import loadImage, saveImage, cudaDeviceSynchronize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('images', nargs=2)
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be between 0 and 1')
    images = [Path(name) for name in args.images]
    if any(not path.is_file() for path in images):
        parser.error('one or both image files do not exist')
    if images[0].stem == images[1].stem:
        parser.error('the two images must have different names')

    output = Path('results')
    output.mkdir(exist_ok=True)
    network = detectNet('ssd-mobilenet-v2', threshold=args.threshold)
    threshold_name = ('%g' % args.threshold).replace('.', 'p')

    for path in images:
        image = loadImage(str(path))
        detections = network.Detect(image, overlay='box,labels,conf')
        cudaDeviceSynchronize()
        stem = '{}_t{}'.format(path.stem, threshold_name)
        image_file = output / (stem + '.png')
        text_file = output / (stem + '.txt')
        saveImage(str(image_file), image)

        lines = [
            'Image: {}'.format(path.name),
            'Threshold: {}'.format(args.threshold),
            'Number of detected objects: {}'.format(len(detections)),
        ]
        for index, item in enumerate(detections, 1):
            left, top = float(item.Left), float(item.Top)
            right, bottom = float(item.Right), float(item.Bottom)
            width, height = right - left, bottom - top
            center = ((left + right) / 2, (top + bottom) / 2)
            lines.extend([
                '',
                'Detection {}'.format(index),
                'ClassID: {}'.format(item.ClassID),
                'Class: {}'.format(network.GetClassDesc(item.ClassID)),
                'Confidence: {:.4f}'.format(item.Confidence),
                'Left: {:.2f}'.format(left),
                'Top: {:.2f}'.format(top),
                'Right: {:.2f}'.format(right),
                'Bottom: {:.2f}'.format(bottom),
                'Width: {:.2f}'.format(width),
                'Height: {:.2f}'.format(height),
                'Area: {:.2f}'.format(width * height),
                'Center: ({:.2f}, {:.2f})'.format(*center),
            ])
        text = '\n'.join(lines) + '\n'
        text_file.write_text(text, encoding='utf-8')
        print(text)
        print('Saved: {} and {}'.format(image_file, text_file))


if __name__ == '__main__':
    main()
