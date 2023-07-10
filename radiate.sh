CUDA_VISIBLE_DEVICES=1 \
python start.py -f processed_/ \
--data_root /data/Radiate/city_3_2 \
--checkpoint ./models/sam_vit_b_01ec64.pth \
--model_type vit_b \
--dataset radiate
