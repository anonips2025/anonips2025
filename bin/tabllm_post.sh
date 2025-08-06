for dataset in albert bank blood calhousing compas covertype credit_card_default creditg diabetes electricity eye_movements heart income jungle road_safety; do
    for numshot in 4 8 16 32 64 128 256; do
        src="/hdd/hans/t-few/exp_out/t03b_${dataset}_synthetic_numshot${numshot}_seed0_ia3_pretrained100k/t0.p"
        dest="eval_res/tabllm/${dataset}/${numshot}_shot/t0.p"
        mkdir -p "eval_res/tabllm/${dataset}/${numshot}_shot"
        cp "$src" "$dest"
        # Extract y_pred and X_synth
        python src/utils/extract_tabllm_inference.py --dataset "$dataset" --numshot "$numshot"
    done
done