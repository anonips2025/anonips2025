for dataset in albert compas covertype credit_card_default electricity eye_movements road_safety; do
    cp "tabllm/templates_${dataset}.yaml" "/hdd/hans/TabLLM/templates/"
done

for dataset in albert bank blood calhousing compas covertype credit_card_default creditg diabetes electricity eye_movements heart income jungle road_safety; do
    cp -r "tabllm/datasets_serialized/${dataset}/" "/hdd/hans/TabLLM/datasets_serialized/${dataset}/"
    cp -r "tabllm/datasets_serialized/${dataset}_synthetic/" "/hdd/hans/TabLLM/datasets_serialized/${dataset}_synthetic/"
done

cp /hdd/hans/kdd25_test/t-few/bin/zero-shot-pretrained-100k.sh /hdd/hans/t-few/bin/
cp /hdd/hans/kdd25_test/t-few/bin/few-shot-pretrained-100k.sh /hdd/hans/t-few/bin/
cp /hdd/hans/kdd25_test/t-few/configs/* /hdd/hans/t-few/configs/
cp /hdd/hans/kdd25_test/t-few/src/models/EncoderDecoder.py /hdd/hans/t-few/src/models/
cp /hdd/hans/kdd25_test/t-few/src/data/* /hdd/hans/t-few/src/data/
cp /hdd/hans/kdd25_test/t-few/src/scripts/get_result_table.py /hdd/hans/t-few/src/scripts/