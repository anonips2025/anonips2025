for dataset in albert bank blood calhousing compas covertype credit_card_default creditg diabetes electricity eye_movements heart income jungle road_safety; do
    for numshot in 4 8 16 32 64 128 256; do
        # Copy synthetic results
        src="/hdd/hans/t-few/exp_out/t03b_${dataset}_synthetic_numshot${numshot}_seed0_ia3_pretrained100k/t0.p"
        dest="eval_res/tabllm/${dataset}/${numshot}_shot/t0.p"
        mkdir -p "eval_res/tabllm/${dataset}/${numshot}_shot"
        cp "$src" "$dest"
        
        # Copy test results
        test_src="/hdd/hans/t-few/exp_out/t03b_${dataset}_test_numshot${numshot}_seed0_ia3_pretrained100k/t0.p"
        test_dest="eval_res/tabllm/${dataset}/${numshot}_shot/t0_test.p"
        cp "$test_src" "$test_dest"
        
        # Extract both synthetic and test data, and calculate metrics
        python src/utils/extract_tabllm_inference.py --dataset "$dataset" --numshot "$numshot" --mode full
    done
done