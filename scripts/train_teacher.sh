GPU="2"
TASK_NAME=cnews
MODEL=hfl/chinese-bert-wwm-ext # teacher models 
DATA_DIR=data/cnews
for TASK_NAME in cnews 
do
for seed in 1 
do 
for LBL_IDX in -1 0 1  # -1: indicate all labels, 0: indicate the first half, 1: indicate the second half 
do 
OUTPUT_DIR=ckpts/${TASK_NAME}-bert-base-${LBL_IDX} #-$seed

CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
  --model_name_or_path $OUTPUT_DIR  --label_index $LBL_IDX \
  --train_file $DATA_DIR/cnews.train.csv --validation_file $DATA_DIR/cnews.val.csv --test_file $DATA_DIR/cnews.test.csv  \
  --fp16  --seed $seed \
  --do_eval  --do_predict  \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --save_total_limit 1 \
  --logging_steps 200  \
  --evaluation_strategy  epoch  \
  --logging_steps 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.1 \
  --output_dir $OUTPUT_DIR
done 
done 
done  