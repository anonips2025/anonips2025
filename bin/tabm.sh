cuda_device="0,1,2,3"

for dataset in albert bank blood calhousing compas covertype credit_card_default creditg diabetes electricity eye_movements heart income jungle road_safety; do
    for numshot in 4 8 16 32 64 128 256; do
        CUDA_VISIBLE_DEVICES="$cuda_device" \
        python src/tabm_eval.py --dataset "$dataset" --numshot "$numshot"
    done
done