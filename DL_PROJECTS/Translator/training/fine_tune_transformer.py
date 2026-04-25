import pandas as pd

from datasets import Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"

df = pd.read_csv("data/english_hindi.csv")

dataset = Dataset.from_pandas(df)

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

def preprocess(example):

    inputs = tokenizer(
        example["english"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    targets = tokenizer(
        example["hindi"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    inputs["labels"] = targets["input_ids"]

    return inputs


dataset = dataset.map(preprocess)

training_args = Seq2SeqTrainingArguments(
    output_dir="model",
    num_train_epochs=2,
    per_device_train_batch_size=8
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("../backend/model")
tokenizer.save_pretrained("../backend/model")