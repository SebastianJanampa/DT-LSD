# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]

        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        lines = torch.cat([t['lines'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_lines = lines.repeat(2 * dn_number, 1)

        known_labels_expaned = known_labels.clone()
        known_lines_expand = known_lines.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(lines))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(lines) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(lines)

        
        known_lines_ = known_lines.clone()
        known_lines_[:, :2] = (known_lines[:, :2] - known_lines[:, 2:]) / 2
        known_lines_[:, 2:] = (known_lines[:, :2] + known_lines[:, 2:]) / 2

        centers = torch.zeros_like(known_lines)
        centers[:, :2] = (known_lines[:, :2] + known_lines[:, 2:]) / 2
        centers[:, 2:] = (known_lines[:, :2] + known_lines[:, 2:]) / 2

        # Noisy length
        diff = torch.zeros_like(known_lines)
        diff[:, :2] = (known_lines[:, 2:] -  known_lines[:, :2]) / 2
        diff[:, 2:] = (known_lines[:, 2:] -  known_lines[:, :2]) / 2

        rand_sign = torch.randint(low=0, high=2, size=(known_lines.shape[0], 2), dtype=torch.float32, device=known_lines.device) * 2.0 - 1.0
        rand_part = torch.rand(size=(known_lines.shape[0], 2), device=known_lines.device)
        rand_part[negative_idx] += 1 
        rand_part *= rand_sign

        known_lines_ = known_lines_ + torch.mul(rand_part.repeat_interleave(2, 1),
                                              diff).cuda() * box_noise_scale        

        # Noisy angle
        theta_bound = torch.deg2rad(torch.tensor(22, device=known_lines.device, dtype=torch.float32))
        rand_sign_theta = torch.randint(low=0, high=2, size=(known_lines.shape[0], 1), device=known_lines.device) * 2.0 - 1
        rand_part_theta = torch.rand(size=(known_lines.shape[0], 1), device=known_lines.device) * theta_bound
        rand_part_theta[negative_idx] +=  theta_bound
        rand_part_theta *= rand_part_theta

        x1_, y1_, x2_, y2_ = torch.split(known_lines_, 1, dim=1)
        s, c = torch.sin(rand_part_theta), torch.cos(rand_part_theta)
        center_x = (x1_ + x2_) / 2
        center_y = (y1_ + y2_) / 2
        x1_ -= center_x
        x2_ -= center_x
        y1_ -= center_y
        y2_ -= center_y
        x1_new = x1_ * c - y1_ * s
        y1_new = x1_ * s + y1_ * c
        x2_new = x2_ * c - y2_ * s
        y2_new = x2_ * s + y2_ * c
        x1_new += center_x
        x2_new += center_x
        y1_new += center_y
        y2_new += center_y
        rotated_lines = torch.hstack((x1_new, y1_new, x2_new, y2_new))

        clamped_lines = torch.zeros_like(rotated_lines)
        w = h = 1 # Normalize image
        eps = 1e-16
        for i,line in enumerate(rotated_lines):
            x1, y1, x2, y2 = line
            slope = (y2 - y1) / (x2 - x1 + eps)
            if x1 < 0:
                x1 = 0
                y1 = y2 + (x1 - x2) * slope
            if y1 < 0:
                y1 = 0
                x1 = x2 - (y2 - y1) / slope
            if x2 > w:
                x2 = w
                y2 = y1 + (x2 - x1) * slope
            if y2 > h:
                y2 = h
                x2 = x1 + (y2 - y1) / slope

            clamped_lines[i, :] = torch.tensor([x1, y1, x2, y2])
        known_lines_expand = clamped_lines

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        input_lines_embed = inverse_sigmoid(known_lines_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_lines = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_lines = padding_lines.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()

        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_lines[(known_bid.long(), map_known_indice)] = input_lines_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
    else:

        input_query_label = None
        input_query_lines = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_lines, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_lines': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_lines'] = out
    return outputs_class, outputs_coord


