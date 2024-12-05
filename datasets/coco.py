# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os

import torch
import torch.utils.data
import torchvision

from datasets.data_util import preparing_dataset
import datasets.transforms as T

__all__ = ['build']


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCocoPolysToMask(object):

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno]
 
        lines = [obj["line"] for obj in anno]
        lines = torch.as_tensor(lines, dtype=torch.float32).reshape(-1, 4)

        lines[:, 2:] += lines[:, :2] #xyxy

        lines[:, 0::2].clamp_(min=0, max=w)
        lines[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["lines"] = lines

        target["labels"] = classes
        
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 680, 690, 704, 736, 768, 788, 800]
    max_size = 1333
    test_size = 640
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))
        

    if image_set == 'train':
        return T.Compose([
            T.RandomSelect(
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                ),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            T.ColorJitter(),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([test_size], max_size=max_size),
            normalize,
        ])



    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    mode = 'lines'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' ),
    }

    # add some hooks to datasets
    img_folder, ann_file = PATHS[image_set]

    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    dataset = CocoDetection(img_folder, ann_file, 
            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args)
        )

    return dataset



if __name__ == "__main__":
    # Objects365 Val example
    dataset_o365 = CocoDetection(
            '/path/Objects365/train/',
            "/path/Objects365/slannos/anno_preprocess_train_v2.json",
            transforms=None,
            return_masks=False,
        )
    print('len(dataset_o365):', len(dataset_o365))
