CUDA_VISIBLE_DEVICES=0 \
python start.py -f cropped-256/ \
--data_root /data/Radiate/city_4_0 \
--checkpoint ./models/sam_vit_b_01ec64.pth \
--model_type vit_b \
--dataset radiate