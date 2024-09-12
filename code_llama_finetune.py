from datetime import datetime
import os
import json
import sys
from datasets import Dataset, DatasetDict
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset

local_model_path = "/codellama-7b-hf"
local_data_path = "/dataset/sql_context_dataset.json"
batch_size = 32
per_device_train_batch_size = 16
gradient_accumulation_steps = batch_size // per_device_train_batch_size
ds_config="./deepspeed_config.json"
output_dir = "sql-code-llama"

training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    deepspeed=ds_config, # Path to DeepSpeed config file
    gradient_checkpointing=True,
    warmup_steps=100,
    max_steps=400,
    learning_rate=3e-4,
    bf16 = True,
    logging_steps=10,
    evaluation_strategy="steps", # if val_set_size > 0 else "no",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    output_dir=output_dir,
    save_total_limit=2,
    load_best_model_at_end=False,
    # ddp_find_unused_parameters=False if ddp else None,
    group_by_length=True, # group sequences of roughly the same length together to speed up training
    run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
)

with open(local_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
if isinstance(data, list):
    data_dict = {key: [d[key] for d in data] for key in data[0]}
else:
    data_dict = data
dataset = Dataset.from_dict(data_dict)

dataset_split = dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")
print(train_dataset[3])

tokenizer = AutoTokenizer.from_pretrained(local_model_path)

eval_prompt = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
### Input:
Which Class has a Frequency MHz larger than 91.5, and a City of license of hyannis, nebraska?

### Context:
CREATE TABLE table_name_12 (class VARCHAR, frequency_mhz VARCHAR, city_of_license VARCHAR)

### Response:
"""
# {'question': 'Name the comptroller for office of prohibition', 'context': 'CREATE TABLE table_22607062_1 (comptroller VARCHAR, ticket___office VARCHAR)', 'answer': 'SELECT comptroller FROM table_22607062_1 WHERE ticket___office = "Prohibition"'}
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
{data_point["answer"]}
"""
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16
)

model.gradient_checkpointing_enable()

model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

# put model back into training mode
model.train()

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)

trainer.train()

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

