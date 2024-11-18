
folder=logs/SWIN-MS5-12-epochs
output=$folder/results

epoch=("000" "001" "002" "003" "004" "005" "006" "007" "008" "009" "010" "011")
#epoch=("011")

for ((i=0;i<${#epoch[@]};++i)); do
	mkdir  -p $output/score
	mkdir  -p $output/score/eval-sAP
	mkdir  -p $output/score/eval-fscore
	mkdir  -p $output/score/eval-sAP-york
	mkdir  -p $output/score/eval-fscore-york
	python main.py \
	--output_dir $output -c config/DTLSD/DTLSD_5scale_swin.py --coco_path data/wireframe_processed \
	--eval --resume $folder/checkpoint0${epoch[i]}.pth --benchmark --dataset val --append_word ${epoch[i]} \
	--options dn_scalar=300 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 epochs=12 lr_drop=11 num_classes=2 dn_labelbook_size=2 
	
	python main.py \
	--output_dir $output -c config/DTLSD/DTLSD_5scale_swin.py --coco_path data/york_processed \
	--eval --resume $folder/checkpoint0${epoch[i]}.pth --benchmark --dataset val --append_word ${epoch[i]} \
	--options dn_scalar=300 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 epochs=12 lr_drop=11 num_classes=2 dn_labelbook_size=2 
		
	python evaluation/eval-sAP-wireframe.py $output/benchmark/benchmark_val_${epoch[i]} | tee -a $output/score/eval-sAP/${epoch[i]}.txt

	python evaluation/eval-fscore-wireframe.py $output/benchmark/benchmark_val_${epoch[i]} | tee $output/score/eval-fscore/${epoch[i]}.txt

	python evaluation/eval-sAP-york.py $output/benchmark/benchmark_york_${epoch[i]} | tee $output/score/eval-sAP-york/${epoch[i]}.txt

	python evaluation/eval-fscore-york.py $output/benchmark/benchmark_york_${epoch[i]} | tee $output/score/eval-fscore-york/${epoch[i]}.txt
		
done


