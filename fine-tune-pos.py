import datasets
import numpy as np
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification

conll2003 = datasets.load_dataset('conll2003', trust_remote_code=True)

conll2003

#print(conll2003)

#print(conll2003['train'][0:2])

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

example_text = conll2003['train'][0]

tokenized_input = tokenizer(example_text['tokens'], is_split_into_words = True)

#print(tokenized_input)

tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])

#print('tokens:\t\t', tokens)

word_ids = tokenized_input.word_ids()

#print('word_ids:\t', word_ids)


def tokenize_and_align_labels(example, label_all_tokens = True):
    tokenized_input = tokenizer(example['tokens'], truncation = True, is_split_into_words = True)
    labels = []

    for i, label in enumerate(example['ner_tags']):
        word_ids =  tokenized_input.word_ids(batch_index = i)
        previous_word_id = None

        label_ids = []

        for word_idx in word_ids:
            if word_idx == None:
                label_ids.append(-100)
            elif word_idx == previous_word_id:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_id = word_idx
        labels.append(label_ids)
    tokenized_input['labels'] = labels
    return tokenized_input

q = tokenize_and_align_labels(conll2003['train'][4:5])

#print('q:\t\t', q)

#for token, label in zip(tokenizer.convert_ids_to_tokens(q['input_ids'][0]), q['labels'][0]):
#    print(f"{token: <40} {label} ")

tokenized_dataset = conll2003.map(tokenize_and_align_labels, batched = True)
print('tokenized_dataset[0:1]:\t', tokenized_dataset['train'][0:2])

model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels = 9)

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    'test-ner',
    eval_strategy = 'epoch',
    learning_rate = 2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = datasets.load_metric('segeval')

example = conll2003['train'][0]

label_list = conll2003['train'].features['ner_tags'].feature.names

print('label_list:\t', label_list)

example_labels = [label_list[i] for i in example['ner_tags']]

print('example_labels:\t', example_labels)

labels = [label_list[i] for i in example['ner_tags']]

metric.compute(predictions = [labels], references = [labels])


def compute_metrics(eval_preds):
    pred_logits, labels = eval_preds
    pred_logits = np.argmax(pred_logits, axis = 2)

    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100 for prediction, label in zip(pred_logits, labels)]
    ]

    results = metric.compute[predictions = predictions, reference = true_labels]

    return {
        'precision': results['overall_precision'],
        'recall': results['overall_recall'],
        'f1': results['overall_f1'],
        'accuracy': results['overall_accuracy'],
    }

trainer = Trainer(
    model,
    args,
    train_dataset = tokenized_dataset['train'],
    eval_dataset = tokenized_dataset['validation'],
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)

trainer.train()

model.save_pretrained('ner_model')

tokenizer.save_pretrained('tokenizer')

id2label = {
    str(i) : label for i, label in enumerate(label_list)
}

label2id = {
    label : str(i) for i, label in enumerate(label_list)
}


import jason

config = json.load(open('ner_model/config'))
config['id2label'] = id2label
config['label2id'] = label2id
json.dump(config, open('ner_model/config', 'w'))

config = json.load(open('ner_model/config'))

model_fine_tuned = AutoModelForTokenClassification('ner_model')


from transformers import pipeline

nlp = pipeline('ner', model = model, tokenizer = tokenizer)

example = 'Bill Gates is the Founder of Microsoft'

ner_results = nlp(example)

print(ner_results)