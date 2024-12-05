_base_ = ['coco_transformer.py']

num_classes = 2

lr = 0.0001
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1

batch_size = 2
weight_decay = 0.0001
epochs = 12
lr_drop = 11
save_checkpoint_interval = 1
clip_max_norm = 0.1

modelname = 'dtlsd'
frozen_weights = None
backbone = 'swin_L_384_22k'
use_checkpoint = True

dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6

pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
add_pos_value = False
two_stage_type = 'standard'
num_select = 300
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
aux_loss = True

set_cost_class = 2.0
set_cost_lines = 5.0
cls_loss_coef = 1.0
lines_loss_coef = 5.0
interm_loss_coef = 1.0

focal_alpha = 0.25

decoder_sa_type = 'sa' # ['sa', 'ca_label', 'ca_content']
matcher_type = 'HungarianMatcher' # or SimpleMinsumMatcher
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True

# for dn
use_dn = True
dn_number = 300
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = True
dn_labelbook_size = 2

# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0


