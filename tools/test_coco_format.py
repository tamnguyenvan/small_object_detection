import os
import numpy as np
import cv2
from pycocotools.coco import COCO


def main():
    image_dir = 'data/images'
    label_file = 'data/test.json'
    class_file = 'data/classes.txt'

    class_map = {i: x.strip() for i, x in enumerate(open(class_file))}
    colors = [tuple(map(int, np.random.randint(0, 256, 3))) for _ in range(len(class_map))]

    coco = COCO(label_file)
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info['file_name']
        image_path = os.path.join(image_dir, filename)
        print(image_path)
        image = cv2.imread(image_path)

        ann_ids = coco.getAnnIds(img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            bbox = ann['bbox']
            class_id = ann['category_id']
            class_name = class_map[class_id-1]

            x1, y1, w, h = bbox
            color = colors[class_id-1]
            cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(image, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
        
        cv2.imshow('img', image)
        cv2.waitKey(0)
        break


if __name__ == '__main__':
    main()