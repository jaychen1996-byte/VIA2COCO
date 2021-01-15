import os
import cv2
import datetime
import json
from utils import getArea
from tqdm import tqdm


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }
    return image_info


def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box, segmentation):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,  # float
        "bbox": bounding_box,  # [x,y,width,height]
        "segmentation": segmentation  # [polygon]
    }
    return annotation_info


def get_segmenation(coord_x, coord_y):
    seg = []
    for x, y in zip(coord_x, coord_y):
        seg.append(x)
        seg.append(y)
    return [seg]


def convert(imgdir, annpath, mode):
    '''
    :param imgdir: directory for your images
    :param annpath: path for your annotations
    :return: coco_output is a dictionary of coco style which you could dump it into a json file
    as for keywords 'info','licenses','categories',you should modify them manually
    '''
    coco_output = {}
    coco_output['info'] = {
        "description": "Nuts Dataset",
        "url": 'http://cocodataset.org',
        "version": "1.0",
        "year": 2020,
        "contributor": "Jay Chen",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }
    coco_output['licenses'] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    if mode == 0:
        coco_output['categories'] = [
            {
                'id': 1,
                'name': 'almond',
                'supercategory': 'nuts',
            },
            {
                'id': 2,
                'name': 'walnut',
                'supercategory': 'nuts',
            },
            {
                'id': 3,
                'name': 'hazelnut',
                'supercategory': 'nuts',
            },
            {
                'id': 4,
                'name': 'cashew',
                'supercategory': 'nuts',
            },
            {
                'id': 5,
                'name': 'pistachio',
                'supercategory': 'nuts',
            },
            {
                'id': 6,
                'name': 'macadamia',
                'supercategory': 'nuts',
            }
        ]
    else:
        coco_output['categories'] = [
            {
                'id': 1,
                'name': 'blueberry',
                'supercategory': 'nuts',
            }
        ]

    coco_output['images'] = []
    coco_output['annotations'] = []

    ann = json.load(open(annpath))
    ann_id = 0
    for img_id, key in tqdm(enumerate(ann.keys())):

        filename = ann[key]['filename']
        # print(filename)
        img = cv2.imread(os.path.join(imgdir, filename))
        image_info = create_image_info(img_id, os.path.basename(filename), img.shape[:2])
        coco_output['images'].append(image_info)
        regions = ann[key]["regions"]
        for region in regions:
            cat = region['region_attributes']['name']  # 我的返回的子类的编号，所以其实不需要这一段，
            # print(cat)
            assert cat in ['扁桃仁', '核桃仁', '榛子', '腰果', '开心果',
                           '夏威夷果', "蓝莓"]
            if cat == '扁桃仁' or cat == "蓝莓":
                cat_id = 1
            elif cat == '核桃仁':
                cat_id = 2
            elif cat == '榛子':
                cat_id = 3
            elif cat == '腰果':
                cat_id = 4
            elif cat == '开心果':
                cat_id = 5
            else:
                cat_id = 6
            iscrowd = 0
            points_x = region['shape_attributes']['all_points_x']
            points_y = region['shape_attributes']['all_points_y']
            area = getArea.GetAreaOfPolyGon(points_x, points_y)
            min_x = min(points_x)
            max_x = max(points_x)
            min_y = min(points_y)
            max_y = max(points_y)
            box = [min_x, min_y, max_x - min_x, max_y - min_y]  # top_left_x,top_left_y,w,h
            segmentation = get_segmenation(points_x, points_y)
            ann_info = create_annotation_info(ann_id, img_id, cat_id, iscrowd, area, box, segmentation)
            coco_output['annotations'].append(ann_info)
            ann_id = ann_id + 1
    return coco_output


if __name__ == '__main__':
    img_path = '/home/jaychen/Desktop/DATASETS/NUTS/train'
    anno_path = './jsonFiles/nuts_485.json'
    result_path = './jsonFiles/instances_train2017.json'
    mode = 0  # 0 nuts 1 blueberry
    result = convert(img_path, anno_path, mode)
    with open(result_path, 'w') as file_obj:
        json.dump(result, file_obj)

    # l1 = json.load(open("via2coco.json", "r"))
    # l2 = json.load(open("instances_val2017.json", "r"))
    # pass
