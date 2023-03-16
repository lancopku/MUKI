GPU="1"
MODEL=hfl/chinese-bert-wwm-ext  
DATA_DIR=data/cnews
TEACHER_WEIGHT='hard'
LR=2e-5
EPOCH=1



for seed in 1 #2 3 1234 # 1234 
do

for TASK in cnews  
do

T1_MODEL=ckpts/$TASK-bert-base-0
T2_MODEL=ckpts/$TASK-bert-base-1


OUTPUT_DIR=result_md/$TASK-bert-base-$seed

CUDA_VISIBLE_DEVICES=$GPU python3 run_mc.py  --learning_rate $LR  --fp16 \
  --t1_model_name_or_path $T1_MODEL --t2_model_name_or_path $T2_MODEL   \
  --student_model_name_or_path $MODEL  --task_name $TASK \
   --seed $seed   \
   --do_train  --train_file $DATA_DIR/cnews.train.csv --validation_file $DATA_DIR/cnews.val.csv --test_file $DATA_DIR/cnews.test.csv \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 --logging_dir $OUTPUT_DIR  --overwrite_output_dir \
  --per_device_eval_batch_size 64 \
  --save_total_limit 1 \
  --evaluation_strategy no  \
  --logging_steps 20 --patience 40 \
  --num_train_epochs $EPOCH \
  --warmup_ratio 0.1 \
  --output_dir $OUTPUT_DIR
done
done 

