from collections import defaultdict
import os, sys, itertools, json
import datasets
from datasets import load_dataset, load_metric
import numpy as np
import nltk
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import logging
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    BartForConditionalGeneration,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
logger = logging.getLogger(__name__)


# classes for argument parsing on running code
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    gen_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    sum_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
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
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

# argument parsing
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# set up logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# set seed
set_seed(training_args.seed)

# load dataset
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file
extension = (
    data_args.train_file.split(".")[-1]
    if data_args.train_file is not None
    else data_args.validation_file.split(".")[-1]
)
raw_datasets = load_dataset(
    extension,
    data_files=data_files,
    delimiter="\t"
)

if training_args.do_train:
    column_names = raw_datasets["train"].column_names
else:
    column_names = raw_datasets["validation"].column_names

# set tokenizer
tokenizer_kwargs = {
    "cache_dir": model_args.cache_dir,
    "use_fast": model_args.use_fast_tokenizer,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}

if model_args.gen_model_name_or_path:
    gen_tokenizer = AutoTokenizer.from_pretrained(model_args.gen_model_name_or_path, **tokenizer_kwargs)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )
if model_args.sum_model_name_or_path:
    sum_tokenizer = AutoTokenizer.from_pretrained(model_args.sum_model_name_or_path, **tokenizer_kwargs)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )

# add special tokens for padding
gen_tokenizer.pad_token = gen_tokenizer.eos_token
sum_tokenizer.pad_token = sum_tokenizer.eos_token

# define tokenize function for preprocessing datasets
def gen_tokenize_function(examples):
    model_inputs = gen_tokenizer(examples['[OUTLINE]'], max_length=1024, return_tensors='pt', padding="max_length", truncation=True)
    with gen_tokenizer.as_target_tokenizer():
        labels = gen_tokenizer(examples['[PLOT]'], max_length=1024, return_tensors='pt', padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def sum_tokenize_function(examples):
    model_inputs = sum_tokenizer(examples['[PLOT]'], max_length=1024, return_tensors='pt', padding="max_length", truncation=True)
    with sum_tokenizer.as_target_tokenizer():
        labels = sum_tokenizer(examples['[OUTLINE]'], max_length=1024, return_tensors='pt', padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# set tokenized datasets
gen_tokenized_datasets = raw_datasets.map(
    gen_tokenize_function,
    #batched=True,
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on dataset",
)
sum_tokenized_datasets = raw_datasets.map(
    sum_tokenize_function,
    #batched=True,
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on dataset",
)

# initialize language models
config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.gen_model_name_or_path and model_args.sum_model_name_or_path:
    gen_config = AutoConfig.from_pretrained(model_args.gen_model_name_or_path, **config_kwargs)
    sum_config = AutoConfig.from_pretrained(model_args.sum_model_name_or_path, **config_kwargs)
else:
    gen_config = CONFIG_MAPPING[model_args.model_type]()
    sum_config = CONFIG_MAPPING[model_args.model_type]()

gen_model = AutoModelForCausalLM.from_pretrained(
    model_args.gen_model_name_or_path,
    from_tf=bool(".ckpt" in model_args.gen_model_name_or_path),
    config=gen_config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
sum_model = BartForConditionalGeneration.from_pretrained(
    model_args.sum_model_name_or_path,
    from_tf=bool(".ckpt" in model_args.sum_model_name_or_path),
    config=sum_config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

# use rouge as main metric
metric = load_metric("rouge")

# define rouge function
def gen_compute_metrics(eval_preds):
    predictions, labels = eval_preds

    decoded_preds = gen_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, gen_tokenizer.pad_token_id)
    decoded_labels = gen_tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != gen_tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def sum_compute_metrics(eval_preds):
    predictions, labels = eval_preds

    decoded_preds = sum_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, sum_tokenizer.pad_token_id)
    decoded_labels = sum_tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != gen_tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# prepare datasets and optimizers
gen_tokenized_datasets.set_format("torch")
sum_tokenized_datasets.set_format("torch")

gen_train_dataloader = DataLoader(gen_tokenized_datasets['train'], batch_size=training_args.per_device_train_batch_size)
gen_eval_dataloader = DataLoader(gen_tokenized_datasets['validation'], batch_size=training_args.per_device_eval_batch_size)
sum_train_dataloader = DataLoader(sum_tokenized_datasets['train'], batch_size=training_args.per_device_eval_batch_size)
sum_eval_dataloader = DataLoader(sum_tokenized_datasets['validation'], batch_size=training_args.per_device_eval_batch_size)

optimizers = []
gen_optimizer = AdamW(itertools.chain(gen_model.parameters(), sum_model.parameters(), gen_model.parameters()), lr=training_args.learning_rate)
#gen_optimizer = AdamW(gen_model.parameters(), lr=training_args.learning_rate)
optimizers.append(gen_optimizer)
#sum_optimizer = AdamW(itertools.chain(sum_model.parameters(), gen_model.parameters(), sum_model.parameters()), lr=training_args.learning_rate)
#sum_optimizer = AdamW(itertools.chain(gen_model.parameters(), sum_model.parameters()), lr=training_args.learning_rate)
optimizers.append(sum_optimizer)
num_epochs = int(training_args.num_train_epochs)
num_training_steps = int(num_epochs * len(gen_train_dataloader))
schedulers = [get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
) for optimizer in optimizers]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
gen_model.to(device)
sum_model.to(device)

progress_bar = tqdm(range(num_training_steps))

# start training
gen_model.train()

# freezing bart
#sum_model.train()
for param in sum_model.parameters():
    param.requires_grad = False

for epoch in range(num_epochs):
# load two datasets in the same order
    gen_iterator = iter(gen_train_dataloader)
    for sums in sum_train_dataloader:
        try:
            gens = next(gen_iterator)
        except StopIteration:
            gen_iterator = iter(gen_train_dataloader)
            gens = next(gen_iterator)
        gens = {k: v[0].to(device) for k, v in gens.items()}
        sums = {k: v[0].to(device) for k, v in sums.items()}

        # first: outline -> plot generation
        gen_outputs = gen_model(**gens)
        gen_first_loss = gen_outputs.loss

        # second: predicted plot -> outline summarization
        gen_pred = torch.argmax(gen_outputs.logits, dim=-1)
        #gen_pred = gen_pred.cpu().detach().numpy()
        gen_pred_sent = gen_tokenizer.batch_decode(gen_pred, skip_special_tokens=True)
        sum_inputs = sum_tokenizer(gen_pred_sent, return_tensors='pt', max_length=1024, padding="max_length", truncation=True)
        sum_inputs['labels'] = sums['labels']
        sum_inputs.to(device)
        sum_outputs = sum_model(**sum_inputs)
        sum_loss = sum_outputs.loss

        # third: predicted outline -> plot generation
        sum_pred = torch.argmax(sum_outputs.logits, dim=-1)
        #sum_pred = sum_pred.cpu().detach().numpy()
        sum_pred_sent = sum_tokenizer.batch_decode(sum_pred, skip_special_tokens=True)
        gen_inputs = gen_tokenizer(sum_pred_sent, return_tensors='pt', max_length=1024, padding='max_length', truncation=True)
        gen_inputs['labels'] = gens['labels']
        gen_inputs.to(device)
        gen_outputs = gen_model(**gen_inputs)
        gen_second_loss = gen_outputs.loss

        # backward
        gen_optimizer.zero_grad()
        gen_loss = gen_first_loss + sum_loss + gen_second_loss
        gen_loss.backward()
        gen_optimizer.step()
        #sum_optimizer.zero_grad()
        #sum_loss.backward()
        #sum_optimizer.step()

        progress_bar.update(1)

logger.info('final gen_loss:{}'.format(gen_loss))
logger.info('final sum_loss:{}'.format(sum_loss))
gen_model.save_pretrained(save_directory='./gen_model_bf')
sum_model.save_pretrained(save_directory='./sum_model_bf')

# start evaluation
gen_model.eval()
sum_model.eval()
gen_evals = []
sum_evals = []

gen_iterator = iter(gen_eval_dataloader)
for sums in sum_eval_dataloader:
    try:
        gens = next(gen_iterator)
    except StopIteration:
        gen_iterator = iter(gen_eval_dataloader)
        gens = next(gen_iterator)
    gens = {k: v[0].to(device) for k, v in gens.items()}
    sums = {k: v[0].to(device) for k, v in sums.items()}

    with torch.no_grad():
        # outline -> plot generation
        gen_outputs = gen_model(**gens)
        # plot -> outline summaraztion
        gen_pred = torch.argmax(gen_outputs.logits, dim=-1)
        sum_inputs = sum_tokenizer(gen_tokenizer.batch_decode(gen_pred, skip_special_tokens=True), return_tensors='pt', max_length=1024, padding="max_length", truncation=True)
        sum_inputs['labels'] = sums['labels']
        sum_inputs.to(device)
        sum_outputs = sum_model(**sum_inputs)
        sum_pred = torch.argmax(sum_outputs.logits, dim=-1)

    # calculate rouge score
    gen_evals.append(gen_compute_metrics((gen_pred.cpu().detach().numpy(), gens['labels'].cpu().detach().numpy())))
    sum_evals.append(sum_compute_metrics((sum_pred.cpu().detach().numpy(), sums['labels'].cpu().detach().numpy())))

# calculate average rouge score
total_gen_evals = defaultdict(float)
total_sum_evals = defaultdict(float)
for gen_eval in gen_evals:
    for k in gen_eval:
        total_gen_evals[k] += gen_eval[k]
for sum_eval in sum_evals:
    for k in sum_eval:
        total_sum_evals[k] += sum_eval[k]
for k in total_gen_evals:
    total_gen_evals[k] /= len(gen_evals)
for k in total_sum_evals:
    total_sum_evals[k] /= len(sum_evals)

# dump rouge score to json files
json.dump(total_gen_evals, open('json/gen_eval_bf.json', 'w'))
json.dump(total_sum_evals, open('json/sum_eval_bf.json', 'w'))
