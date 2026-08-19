#!/bin/bash

# echo "Starting foreground mask extraction for cloth_no_occlusion_back_3sec..."
# for chunk in 0 3 7 12 20; do
#   echo "Processing chunk $chunk..."
#   python obtain_foreground_mask.py \
#     --chunk_path /mnt/mydisk/captured_data_double_arm/cloth_no_occlusion_back_3sec/chunk_${chunk}
# done

# echo "Starting foreground mask extraction for cloth_no_occlusion_back_4sec..."
# for chunk in 8 13; do
#   echo "Processing chunk $chunk..."
#   python obtain_foreground_mask.py \
#     --chunk_path /mnt/mydisk/captured_data_double_arm/cloth_no_occlusion_back_4sec/chunk_${chunk}
# done

# echo "Starting foreground mask extraction for cloth_no_occlusion_front_3sec..."
# for chunk in 2 5 6 7 11 14 17; do
for chunk in 14 17; do
  echo "Processing chunk $chunk..."
  python obtain_foreground_mask.py \
    --chunk_path /mnt/mydisk/captured_data_double_arm/cloth_no_occlusion_front_3sec/chunk_${chunk}
done

echo "Starting foreground mask extraction for cloth_no_occlusion_front_4sec..."
for chunk in 15 21 22 23 27 28; do
  echo "Processing chunk $chunk..."
  python obtain_foreground_mask.py \
    --chunk_path /mnt/mydisk/captured_data_double_arm/cloth_no_occlusion_front_4sec/chunk_${chunk}
done