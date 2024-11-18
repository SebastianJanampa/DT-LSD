
python main.py \
	--output_dir logs/R50-MS5-12-epochs -c config/DINO/DINO_5scale.py --coco_path data/wireframe_processed \
	--resume pretrain/checkpoint_5scale.pth --no_opt --pretrain\
	--options dn_scalar=300 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 epochs=12 lr_drop=11 \
	num_classes=2 dn_labelbook_size=2 focal_alpha=0.1 batch_size=2
