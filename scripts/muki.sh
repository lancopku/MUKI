GPU="1"
MODEL=hfl/chinese-bert-wwm-ext #chinese-bert-base 
DATA_DIR=data/cnews
HARD=0 # hard or soft integration
SCHEDULE=1 # margin-based instance re-weight 
NMC=0 # whether to use Monte-carlo dropout estimated teacher uncertainty, 0: single forward 1: Monte-carlo dropout
TST=0.2   # temperature for soft integration
TEACHER_MC_WEIGHT=$FILE_TO_SAVE_TEACHER_MC_WEIGHT  # see readme for downloading corresponding files 


for TASK in cnews
do

T1_MODEL=ckpts/$TASK-bert-base-0
T2_MODEL=ckpts/$TASK-bert-base-1

for seed in  1 #  2 3 
do
#do 
OUTPUT_DIR=result_uka_m/$TASK-bert-base-$seed-hard${HARD}-schedule${SCHEDULE}-nomc$NMC-TN$TN-TST$TST-base 
CUDA_VISIBLE_DEVICES=$GPU python3 run_uka_m.py   --fp16 --no_mc $NMC  --teacher_number 2 --teacher_score_temperature $TST \
  --teacher_paths "$T1_MODEL;$T2_MODEL" --hard_teacher $HARD --consistency_schedule $SCHEDULE \
  --student_model_name_or_path $MODEL --task_name $TASK --teacher_weight_file $TEACHER_MC_WEIGHT \
   --seed $seed  --train_file $DATA_DIR/cnews.train.csv --validation_file $DATA_DIR/cnews.val.csv --test_file $DATA_DIR/cnews.test.csv  \
  --do_predict  --do_eval --do_train \
  --max_seq_length 128 \
  --per_device_train_batch_size 16\
  --per_device_eval_batch_size 64 \
  --save_total_limit 1 \
  --metric_for_best_model accuracy \
  --load_best_model_at_end \
  --evaluation_strategy  steps  \
  --logging_steps 200 --patience 50 \
  --learning_rate 2e-5 \
  --num_train_epochs 3  \
  --warmup_ratio 0.1 \
  --output_dir $OUTPUT_DIR
done
done  