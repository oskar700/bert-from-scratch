import sys
import os
import json
from datasets import Dataset
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments


vocab_size = 30522
max_length = 512
tokenizer_path = './domain-bert-tokenizer'

# Generates a dictionary with the label as key and a numeric index as value.
# The numeric values range from 0 - n (number of labels) with [UNK] (unknown) label as 0 index.
def numerize_label_types(label_types):
    label_to_id = {}
    unknown_label = '[UNK]'

    id = 0
    label_to_id[unknown_label] = id
    for label in label_types:
        id += 1
        label_to_id[label] = id

    return label_to_id


def train_tokenizer(text):
    vocab_file_name = './vocab_file.txt'
    vocab_file = open(vocab_file_name, 'w')
    for line in text:
        vocab_file.write(line)
        vocab_file.write('\n')
    vocab_file.close()

    tokenizer = BertWordPieceTokenizer(lowercase=True)
    tokenizer.train([vocab_file_name], vocab_size=vocab_size, min_frequency=2, 
                    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.save_model(tokenizer_path)


def train_base_model(training_dataset):
    hf_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    def tokenize_function(examples):
        return hf_tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, padding=True)
    
    block_size = 128
    tokenized_dataset = training_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
 
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        return {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
    tokenized_dataset = tokenized_dataset.map(group_texts, batched=True)
    
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        type_vocab_size=2
    )

    model = BertForMaskedLM(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=hf_tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./domain-bert",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset
    )

    trainer.train()

    model.save_pretrained("./domain-bert")
    hf_tokenizer.save_pretrained("./domain-bert")


def process_training_file(file_name):
    tokens = []
    labels = []
    text = []
    label_types = set()     # Holds the label types: road, street_name, postcode etc.

    print("Loading dataset")
    i = 0
    with open(file_name) as input_file:
        for line in input_file:
            if len(line) == 0:
                continue
            # line is of this format: <language>\t<country>\t<address data with labels (token/label)>
            tsv_elems = line.lower().split('\t')
            # Third element is the actual address data.
            elems = tsv_elems[2].split()
            line_tokens = []
            line_labels = []
            for elem in elems:
                tokens_labels = elem.split('/')

                label = tokens_labels[len(tokens_labels) - 1]
                # Skip separation tokens (| , etc.)
                if 'sep' in label:
                    continue
                token = '/'.join(tokens_labels[0:len(tokens_labels) - 1])

                line_tokens.append(token)
                line_labels.append(label)
                label_types.add(label)

            tokens.append(line_tokens)
            labels.append(line_labels)
            text.append(' '.join(line_tokens))

            i += 1
            if i % 10000000 == 0:
                print("Loaded", i, "records")

    print("Processed total", i, "records")
    # Assign numeric values to token types.
    print("Finished loading data")
    label_to_id = numerize_label_types(label_types)

    # Give numeric value to all the labels found in the training set.
    pos_tags = []
    for label_list in labels:
        pos_list = []
        for label in label_list:
            idx = label_to_id[label]
            pos_list.append(idx)
        pos_tags.append(pos_list)

    raw_dataset = {
#        'tokens' : tokens,
#        'labels-1' : labels,
        'text' : text
#        'pos_tags' : pos_tags
    }

    training_dataset = Dataset.from_dict(raw_dataset)

    print("Traiining tokenizer")
    train_tokenizer(text)

    print("Training base model")
    train_base_model(training_dataset)
    print("Processing complete!!!")


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python3 datasource-own-data.py <input file>")

    file_name = sys.argv[1]

    process_training_file(file_name)
