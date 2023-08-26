CUDA_VISIBLE_DEVICES=1 \
python start.py -f processed_1600/ \
--data_root /data/oxford/2019-01-17-12-48-25-radar-oxford-10k \
--checkpoint ./models/sam_vit_b_01ec64.pth \
--model_type vit_b \
--dataset oxford \
--patch_num 1 \
