import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

import numpy as np
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from no_shuffle_trainer import NoShuffleTrainer
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from models.monte_carlo import MCBert

glue_tasks = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "imdb": ("text", None),
    'boolq': ("passage", "question"),
    'yelp_polarity': ("text", None),
    'yelp_review_full': ("text", None),
    'ag_news': ("text", None),
    'tnews': ('sentence', None),
    'dbpedia_14': ('content', None),
    'cnews': ('sentence', None),
    'abstract': ('sentence', None),
    'gs': ('sentence', None),
    'sogou': ('sentence', None),
    'yahoo_answers_topics': ('sentence', None),  # no use here, actually use preprocess function instead
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    teacher_weight_file: Optional[str] = field(
        default=None, metadata={"help": "teacher weight file"}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    few_shot_k: Optional[int] = field(default=-1,
                                      metadata={"help": "Number of instance for training of each class"})
    split_eval: Optional[bool] = field(default=False,
                                       metadata={"help": "Number of instance for training of each class"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """


    student_model_name_or_path: str = field(default='bert-base-uncased',
                                            metadata={
                                                "help": "Path to the pre-trained lm"}
                                            )
    t1_model_name_or_path: Optional[ str] = field(default=None,
        metadata={"help": "Path to pretrained bert model or model identifier from huggingface.co/models"}
    )
    t2_model_name_or_path:Optional[str] = field(default=None,
        metadata={"help": "Path to pretrained bert model or model identifier from huggingface.co/models"}
    )
    kd_alpha: Optional[float] = field(
        default=1.0, metadata={"help": "loss alpha for kd"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    dropout: Optional[float] = field(default=0.1, metadata={"help": "dropout rate for monte carlo "})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "KD distillation kl temperature"})
    eval_strategy: Optional[str] = field(default='student', metadata={"help": "eval strategy of model"})
    mode: Optional[str] = field(default='auto', metadata={"help": "margin loss weight mode"})
    sts_weight: Optional[float] = field(default=0.0, metadata={"help": "weight of self-boosted probability fusion"})
    relation_weight: Optional[float] = field(default=0.0, metadata={"help": "weight of  margin relation  weight"})
    margin: Optional[float] = field(default=1.0, metadata={"help": "weight of  margin relation  weight"})

    vkd_weight: Optional[float] = field(default=1.0, metadata={"help": "weight of  margin relation  weight"})
    patience: Optional[int] = field(default=50, metadata={"help": "early stop patience"})
    teacher_paths: Optional[str] = field(default=None, metadata={"help": "teacher paths, split by ; "})
    teacher_number: Optional[int] = field(default=2, metadata={"help": "early stop patience"})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.task_name is not None and data_args.task_name in glue_tasks:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    elif data_args.task_name is not None and data_args.task_name in task_to_keys and data_args.task_name not in [
        'tnews', 'cnews', 'abstract', 'sogou', 'gs']:  # other supported tasks
        datasets = load_dataset(data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file}
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files['test'] = data_args.test_file
        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, delimiter='\t')
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)

    # Labels
    if data_args.task_name is not None and data_args.task_name in glue_tasks:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        # hack here, we not support regression now
        is_regression = False  # datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        elif data_args.task_name == 'boolq':
            label_list = ["False", "True"]
            num_labels = 2
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    logger.info("Number of Total Labels: %d" % num_labels)

    logger.info("Teacher Number %d" % (num_labels // model_args.teacher_number))
    label_per_teacher = num_labels // model_args.teacher_number
    teacher_num_labels = []
    for i in range(model_args.teacher_number):
        teacher_label_list = label_list[i * label_per_teacher: (i + 1) * label_per_teacher
        if i != model_args.teacher_number - 1 else None]
        logger.info("Teacher  %d: Num label %d" % (i, len(teacher_label_list)))
        teacher_num_labels.append(len(teacher_label_list))

    teacher_configs = []
    teacher_paths = model_args.teacher_paths.split(";")
    print(teacher_paths)
    assert len(teacher_paths) == model_args.teacher_number
    for i in range(model_args.teacher_number):
        t_config = AutoConfig.from_pretrained(
            teacher_paths[i],
            num_labels=teacher_num_labels[i],
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        t_config.hidden_dropout_prob = model_args.dropout
 
        teacher_configs.append(t_config)

    # assume all models use same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_paths[0],
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print(teacher_paths, teacher_configs)
    teacher_models = []

    for t_path, t_config in zip(teacher_paths, teacher_configs):
    
        t_model = AutoModelForSequenceClassification.from_pretrained(
            t_path,
            from_tf=False,
            config=t_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )     
        logger.info(t_model.device)
        teacher_models.append(t_model)
    # trained teacher model

    student_config = deepcopy(teacher_configs[0])
    student_config.num_labels = num_labels

    model = MCBert.from_pretrained(
        model_args.student_model_name_or_path,
        config=student_config,
        kd_alpha=model_args.kd_alpha,
        teachers=None,
        temperature=model_args.temperature,
        eval_strategy=model_args.eval_strategy,
        margin=model_args.margin,
        mode=model_args.mode
    )
    print(model.device)


    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if 'tnews' in data_args.train_file:
                logger.info("Using tnews dataset")
                sentence1_key, sentence2_key = "sentence", None
            elif len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    if data_args.teacher_weight_file:
        logger.info(f"Loading teacher weight from {data_args.teacher_weight_file}")
        teacher_weight = np.load(data_args.teacher_weight_file)
    else:
        teacher_weight = None

    def preprocess_function(examples, idx):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        if teacher_weight is not None:
            result["teacher_weight"] = teacher_weight[
                idx]  # get corresponding teacher weight, only provided for training data
        return result

    def preprocess_function_yahoo(examples):
        texts = []
        for qt, qc, ba in zip(examples['question_title'], examples['question_content'], examples['best_answer']):
            text = qt + ' ' + qc + ' ' + ba
            texts.append(text)
        result = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True)
        result["label"] = [label_to_id[t] for t in examples["topic"]]
        return result

    datasets = datasets.map(
        preprocess_function if data_args.task_name != 'yahoo_answers_topics' else preprocess_function_yahoo,
        with_indices=True,
        batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]

    if data_args.few_shot_k >= 0:  # few shot setting, only use k instances of each class for training
        logger.info("Shuffling data")
        # shuffle and sample a  num_labels * data_args.few_shot_k class samples for trianing
        split = train_dataset.train_test_split(test_size=1, train_size=num_labels * data_args.few_shot_k,
                                               seed=training_args.seed)
        train_dataset = split['train']


    logger.info("Number of training instances: %d" % len(train_dataset))
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None and data_args.task_name in glue_tasks:
        metric = load_metric("glue", data_args.task_name)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None and data_args.task_name in glue_tasks:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    logger_callback = AKACallBack(model)
    # early_stop_callback = EarlyStoppingCallback(early_stopping_patience=model_args.patience)

    # Initialize our Trainer
    trainer = NoShuffleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[logger_callback]  # , early_stop_callback],
    )
    teachers = [ t.to(trainer.args.device) for t in teacher_models ]
    model.teachers = teachers 
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in sorted(eval_result.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]

        for test_dataset, task in zip(test_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=test_dataset)
            output_eval_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    for key, value in sorted(eval_result.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")
            if data_args.split_eval:
                for label_index in range(2):
                    label_per_teacher = num_labels // 2
                    label_list_to_eval = label_list[label_index * label_per_teacher: (
                                                                                             label_index + 1) * label_per_teacher if label_index == 0 else None]
                    print('evaluating on label subset', label_list_to_eval)
                    logger.info("Len before filtering: %d" % (len(test_dataset)))
                    test_dataset_split = test_dataset.filter(lambda example: example['label'] in label_list_to_eval)
                    eval_result = trainer.evaluate(eval_dataset=test_dataset_split)
                    output_eval_file = os.path.join(training_args.output_dir,
                                                    f"test_results_{task}_split{label_index}.txt")
                    if trainer.is_world_process_zero():
                        with open(output_eval_file, "w") as writer:
                            logger.info(f"***** Eval results {task} *****")
                            for key, value in sorted(eval_result.items()):
                                logger.info(f"  {key} = {value}")
                                writer.write(f"{key} = {value}\n")

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
