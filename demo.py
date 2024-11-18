import os, sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
import torchvision.transforms.functional as functional
import torch.nn.functional as F
from models import build_model
from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image

class ToTensor(object):
    def __call__(self, img):
        return functional.to_tensor(img)

def resize(image, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = functional.resize(image, size)

    return rescaled_image

class Resize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img):
        size = self.sizes
        return resize(img, size, self.max_size)

model_config_path = "config/DINO/DINO_4scale_swin.py" # change the path of the model config file
model_checkpoint_path = "logs/1100/SWIN-MS4-36-epochs/checkpoint0035.pth" # change the path of the model checkpoint

# obtain checkpoints
checkpoint = torch.load(model_checkpoint_path)

# load model
args = checkpoint['args']
model, _, postprocessors = build_model(args)
model.load_state_dict(checkpoint['model'])
model.to('cuda')
model.eval()

PLTOPTS = {"color": "#33FFFF", "s": 6, "edgecolors": "none", "zorder": 5}
# load image
files = ['00031841', '00035228', 'P1080074', 'P1080116']
for file in files:
    try:
        raw_img = Image.open(f'./figures/{file}.png').convert("RGB")
    except:
        raw_img = Image.open(f'./figures/{file}.jpg').convert("RGB")
    w, h = raw_img.size
    orig_size = torch.as_tensor([int(h), int(w)])

    # normalize image
    test_size = 1100
    normalize = Compose([
        ToTensor(),
        Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273]),
        Resize([test_size]),
    ])
    img = normalize(raw_img)
    inputs = nested_tensor_from_tensor_list([img])


    with torch.no_grad():
        outputs = model(inputs.to("cuda"))

    out_logits, out_line = outputs['pred_logits'][0].cpu(), outputs['pred_lines'][0].cpu()
    prob = out_logits.sigmoid()

    scores, labels = prob[..., :-1].max(-1)

    img_h, img_w = orig_size.unbind(0)
    scale_fct = torch.unsqueeze(torch.stack([img_w, img_h, img_w, img_h], dim=0), dim=0)
    lines = out_line #* scale_fct[:, None, :]
    lines = lines.view(900, 2, 2)
    lines = lines.flip([-1])# this is yxyx format
    lines = lines.detach().numpy()
    scores = scores.detach().numpy()
    keep = scores >= 0.25
    keep = keep.squeeze()
    lines = lines[keep]
    lines = lines.reshape(lines.shape[0], -1)
    extent=[0, 1, 1, 0]

    plt.figure(figsize=(5, 5))
    plt.imshow(raw_img, extent=extent)

    for tp_id, line in enumerate(lines):
        y1, x1, y2, x2 = line # this is yxyx
        p1 = (x1, y1)
        p2 = (x2, y2)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=1.5, color='darkorange', zorder=1)
        plt.scatter(p1[0], p1[1], **PLTOPTS)
        plt.scatter(p2[0], p2[1], **PLTOPTS)
    plt.axis([0, w-1, h-1, 0])
    plt.axis('off')
    plt.axis('equal')
    plt.savefig(f'./figures/{file}_VIS.png', pad_inches=0.1, bbox_inches='tight')
    # plt.show()