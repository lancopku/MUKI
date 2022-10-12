
GPU="0"
DATA_DIR=data/cnews
MODEL=hfl/chinese-bert-wwm-ext  #bert-base-uncased #
for seed in 1 
do

for TASK in cnews 
do

T1_MODEL=ckpts/$TASK-bert-base-0
T2_MODEL=ckpts/$TASK-bert-base-1

OUTPUT_DIR=result_cfl/$TASK-bert-base-$seed
CUDA_VISIBLE_DEVICES=$GPU python3 run_dfa.py   --fp16 \
  --t1_model_name_or_path $T1_MODEL --t2_model_name_or_path $T2_MODEL \
  --student_model_name_or_path $MODEL  --task_name $TASK \
  --seed $seed --train_file $DATA_DIR/cnews.train.csv --validation_file $DATA_DIR/cnews.val.csv --test_file $DATA_DIR/cnews.test.csv \
  --do_predict  --do_eval --do_train \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --save_total_limit 1 \
  --metric_for_best_model  accuracy \
  --load_best_model_at_end \
  --evaluation_strategy  steps  \
  --logging_steps 200 \
  --learning_rate 2e-5 \
  --num_train_epochs 3  \
  --warmup_ratio 0.1 \
  --output_dir $OUTPUT_DIR
done 
done 