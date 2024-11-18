
folder=logs/SWIN-MS5-24-epochs
output=$folder/results

epoch=("023")

for ((i=0;i<${#epoch[@]};++i)); do
	python benchmark.py --output_dir $output -c config/DTLSD/DTLSD_5scale_swin.py --coco_path data/wireframe_processed \
	--test --resume $folder/checkpoint0${epoch[i]}.pth --dataset val\
	--options embed_init_tgt=TRUE use_ema=False num_classes=2 dn_labelbook_size=2
done


