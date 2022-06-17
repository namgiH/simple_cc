# Simple experiments for Cycle Consistency
Try 
```
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file data/en_bart_ncc/test_encoded.csv \
    --validation_file data/en_bart_ncc/test_encoded.csv \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir tmp/ \
    --do_train \
    --do_eval \
    --save_strategy epoch \
    --evaluation_strategy epoch   
```
, after `pip install -r reqs.txt --extra-index-url https://download.pytorch.org/whl/cu113`.
