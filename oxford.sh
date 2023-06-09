CUDA_VISIBLE_DEVICES=0 \
python start.py -f processed_/ \
--data_root /data/oxford/2019-01-10-11-46-21-radar-oxford-10k \
--checkpoint ./models/sam_vit_b_01ec64.pth \
--model_type vit_b \
--dataset_name oxford