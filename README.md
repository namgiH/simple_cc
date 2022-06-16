# Simple experiments for Cycle Consistency
Try ```python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file data/en_bart_ncc/test_encoded.csv \
    --validation_file data/en_bart_ncc/test_encoded.csv \
    --per_device_train_batch_size 1 \
    --output_dir tmp/ \
    --do_train```, after install `reqs.txt`.
