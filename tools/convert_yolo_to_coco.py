import os
import glob
import json
import argparse
import numpy as np

import cv2


def load_txt(label_file):
    """Load bounding boxes data from the given label file.

    Parameters
    label_file: str
        Path to label file.
    
    Returns
    data: list
        List of bounding boxes data.
    """
    data = []
    with open(label_file) as f:
        for line in f:
            row = line.strip().split()
            class_id = int(row[0])
            bboxes = list(map(float, row[1:]))
            data.append((class_id, *bboxes))
    return data


def write_coco(data, classes, out_label_path):
    """Write label json file in COCO format.
    
    Parameters
    
    out_label_path: str
        Path to output label json file. 
    """
    image_paths, label_paths = data
    categories = []
    for i, class_name in enumerate(classes):
        categories.append({
            'id': i + 1,
            'name': class_name,
            'supercategory': class_name,
        })

    content = {}
    content['info'] = {
        'description': '',
        'url': '',
        'version': '',
        'year': 2021,
        'contributor': '',
        'date_created': ''
    }
    content['licenses'] = [{'id': 1, 'name': None, 'url': None}]
    content['categories'] = categories
    content['annotations'] = []
    content['images'] = []

    bboxes_cnt = 0
    for image_path, label_path in zip(image_paths, label_paths):
        # Load image info
        image_filename = os.path.basename(image_path)
        image_id = int(image_filename.split('.')[0])
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        # Store image context
        image_context = {
            'file_name': image_filename,
            'height': height,
            'width': width,
            'id': image_id,
        }
        content['images'].append(image_context)

        # Load annotations
        data = load_txt(label_path)
        for class_id, *bboxes in data:
            bbox_context = {}
            bbox_context['id'] = bboxes_cnt
            bbox_context['image_id'] = image_id
            bbox_context['category_id'] = class_id + 1
            bbox_context['iscrowd'] = 0

            x_center_norm, y_center_norm, w_norm, h_norm = bboxes
            x_center, y_center = x_center_norm * width, y_center_norm * height
            w, h = int(w_norm * width), int(h_norm * height)
            xmin = int(x_center - w // 2)
            ymin = int(y_center - h // 2)
            xmax = int(x_center + w // 2)
            ymax = int(x_center + h // 2)
            bbox_context['area']  = h * w
            bbox_context['bbox'] = [xmin, ymin, w, h]
            bbox_context['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
            content['annotations'].append(bbox_context)
            bboxes_cnt += 1
    
    # Save label file
    with open(out_label_path, 'wt') as f:
        json.dump(content, f)
    print(f'Saved as {out_label_path}')


def convert_yolo_to_coco_format(
    data_dir,
    out_dir,
    split,
    extensions=['.jpg', '.png']):
    """Convert YOLO format to COCO format.
    
    Parameters
    data_dir: str
        Path to image directory.
    out_label_path: str
        Path to output label json file.
    extensions: list
        Supported image extensions.
    """
    assert os.path.isdir(data_dir), f'Not found input {data_dir}'

    # Load image paths and labels
    image_paths = []
    for ext in extensions:
        image_paths += list(glob.glob(os.path.join(data_dir, f'*{ext}')))
    label_paths = [os.path.splitext(path)[0] + '.txt' for path in image_paths]
    assert len(image_paths) == len(label_paths), 'Number of images and labels must match'
    print(f'Found {len(image_paths)} samples')

    splits = {}
    if split > 0:
        num_train = int(len(image_paths) * (1 - split))
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)
        image_paths = np.array(image_paths)[indices].tolist()
        label_paths = np.array(label_paths)[indices].tolist()

        train_image_paths = image_paths[:num_train]
        train_label_paths = label_paths[:num_train]
        test_image_paths = image_paths[num_train:]
        test_label_paths = label_paths[num_train:]
        print('Split the origin dataset into training and testing set')
        print('Number of training samples:', len(train_image_paths))
        print('Number of testing samples:', len(test_image_paths))

        splits = {
            'train': [train_image_paths, train_label_paths],
            'test': [test_image_paths, test_label_paths],
        }
    else:
        splits = {'train': [image_paths, label_paths]}
    
    # Check class file
    class_file = os.path.join(data_dir, '..', 'classes.txt')
    assert os.path.isfile(class_file), f'Not found class file {class_file}'

    # Load class file
    classes = [x.strip() for x in open(class_file)]

    for split, data in splits.items():
        print(f'Creating dataset {split}...')
        out_label_path = os.path.join(out_dir, f'{split}.json')
        write_coco(data, classes, out_label_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='Path to data directory.')
    parser.add_argument('--out-dir', type=str, help='Path to output directory.')
    parser.add_argument('--split', type=float, default=0., help='Split percentage.')
    args = parser.parse_args()
    print(args)

    convert_yolo_to_coco_format(args.data_dir, args.out_dir, args.split)