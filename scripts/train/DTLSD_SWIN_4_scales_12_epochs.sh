
 python main.py \
	--output_dir logs/SWIN-MS4-12-epochs -c config/DTLSD/DTLSD_4scale_swin.py --coco_path data/wireframe_processed \
	--resume pretrain/checkpoint_4scale_swin.pth --no_opt --pretrain\
	--options dn_number=300 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 epochs=12 lr_drop=11 \
	num_classes=2 dn_labelbook_size=2 focal_alpha=0.1 batch_size=2
