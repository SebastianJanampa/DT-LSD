# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Benchmark inference speed of Deformable DETR.
"""
import os
import time
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from util.misc import nested_tensor_from_tensor_list
from util.slconfig import DictAction, SLConfig
import util.misc as utils


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    parser.add_argument('--benchmark', action='store_true',
                        help="Train segmentation head if the flag is provided")
    #parser.add_argument('--batch_size', default=2, type=int)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='data/wireframe_processed')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    parser.add_argument('--dataset', default='train', type=str, choices=('train', 'val'))
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--no_opt', action='store_true')
    parser.add_argument('--append_word', default=None, type=str, help="Name of the convolutional backbone to use")

    return parser

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

@torch.no_grad()
def measure_average_inference_time(model, data_loader, device, warm_iters=10):
    ts = []
    for iter_, (samples, _) in enumerate(data_loader):
        samples = samples.to(device)
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        model(samples)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        if iter_ >= warm_iters:
          ts.append(t)
    return sum(ts) / len(ts)


def benchmark(args):

    device = torch.device(args.device)


    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))


    # Model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

    model.eval()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    # Dataset
    dataset = build_dataset(image_set=args.dataset, args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(dataset, 1, sampler=sampler,
                             drop_last=False, 
                             collate_fn=utils.collate_fn, 
                             num_workers=2)


    # if args.resume is not None:
    #     ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    #     model.load_state_dict(ckpt['model'])
    t = measure_average_inference_time(model, data_loader, device)
    return 1.0 * 1 / t 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    fps = benchmark(args)
    print(f'Inference Speed: {fps:.1f} FPS')

