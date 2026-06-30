from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import resample
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModel, TrainingArguments, Trainer
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import torch.optim as optim
from evaluate import load
from sklearn.metrics import classification_report, confusion_matrix
from detoxify import Detoxify

def read_data(file_path="tCF Review Data - reviews_last_8y.csv"):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df = df.dropna(subset=['labelling (hurtful, not hurtful, hurtful but useful)'])
    df = df.reset_index(drop=True)
    
    label_corrections = {
        'hurtful but helpful': 'hurtful but useful',
        'hurtful but userful': 'hurtful but useful',
        'not hrutful': 'not hurtful',
        'not hurtftul': 'not hurtful'
    }
    
    df['labelling (hurtful, not hurtful, hurtful but useful)'] = df['labelling (hurtful, not hurtful, hurtful but useful)'].replace(label_corrections)
    
    label_string_to_num = {
        'not hurtful': 0,
        'hurtful but useful': 1,
        'hurtful': 2
    }

    df['labelling (hurtful, not hurtful, hurtful but useful)'] = df['labelling (hurtful, not hurtful, hurtful but useful)'].replace(label_string_to_num)

    X = df[['instructor_rating', 'recommendability', 'enjoyability', 'difficulty', 'hours_per_week', 'text']]
    y = df['labelling (hurtful, not hurtful, hurtful but useful)']

    # train vs test (20% test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # train vs val (25% of remaining so 60/20/20 total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=42
    )

    train_df = X_train.copy()
    train_df["label"] = y_train.values

    df_0 = train_df[train_df["label"] == 0]  # not hurtful
    df_1 = train_df[train_df["label"] == 1]  # hurtful but useful
    df_2 = train_df[train_df["label"] == 2]  # hurtful

    print("Before oversampling:", len(df_0), len(df_1), len(df_2))

    target_size = len(df_0)

    df_1_up = resample(df_1, replace=True, n_samples=target_size, random_state=42)
    df_2_up = resample(df_2, replace=True, n_samples=target_size, random_state=42)

    train_df_balanced = pd.concat([df_0, df_1_up, df_2_up])
    train_df_balanced = train_df_balanced.sample(frac=1.0, random_state=42)  # shuffle

    print("After oversampling:", train_df_balanced["label"].value_counts())

    X_train = train_df_balanced.drop(columns=["label"])
    y_train = train_df_balanced["label"]

    return X_train, y_train, X_val, y_val, X_test, y_test

def tokenize_data(X_train, X_test, tokenizer_name="distilbert-base-uncased-finetuned-sst-2-english"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenized_train = tokenizer(X_train['text'].tolist(), padding="max_length", truncation=True,max_length=128, return_tensors="pt")

    train_input_ids = tokenized_train['input_ids']
    train_attention_mask = tokenized_train['attention_mask']
    train_metadata = torch.from_numpy(X_train[['instructor_rating', 'recommendability', 'enjoyability', 'difficulty', 'hours_per_week']].values)
    train_metadata = train_metadata.float()

    tokenized_test = tokenizer(X_test['text'].tolist(), padding="max_length", truncation=True,max_length=128, return_tensors="pt")

    test_input_ids = tokenized_test['input_ids']
    test_attention_mask = tokenized_test['attention_mask']
    test_metadata = torch.from_numpy(X_test[['instructor_rating', 'recommendability', 'enjoyability', 'difficulty', 'hours_per_week']].values)
    test_metadata = test_metadata.float()

    y_train = torch.from_numpy(np.array(y_train)).long()
    y_test = torch.from_numpy(np.array(y_test)).long()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return train_input_ids, train_attention_mask, train_metadata, y_train, test_input_ids, test_attention_mask, test_metadata, y_test, data_collator

# also could potentially use dropout layers

class CustomBERTModel(nn.Module):
    def __init__(self, pretrained_model_name, num_metadata, num_labels, class_weights=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.metadata_model = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + num_metadata, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, metadata=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_embeddings = output.last_hidden_state[:,0,:]
        cls_with_metadata = torch.cat([cls_token_embeddings, metadata],dim=1)
        logits = self.metadata_model(cls_with_metadata)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return {"loss": loss, "logits": logits}

def class_imbalance_weights(train_metadata, y_train, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    #trying to fix class imbalance, very few "hurtful" and "hurtful but useful" review texts
    labels = y_train.numpy() # Convert y_train tensor to numpy array
    base_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)

    # boost minority classes further
    scale = np.array([1.0, 1.5, 2.0])  # tweak these
    class_weights = base_weights * scale

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    print("class weights:", class_weights_tensor)

    #init custom model w/ class weights
    custom_model = CustomBERTModel(
        pretrained_model_name=model_name,
        num_metadata=train_metadata.shape[1],
        num_labels=3,
        class_weights=class_weights_tensor
    )

    return custom_model
    
def freeze_bert_layers(custom_model):
    # Freeze the first num_layers_to_freeze layers of the BERT model
    for name, param in custom_model.bert.named_parameters():
        if name.startswith("encoder.layer.8") or \
        name.startswith("encoder.layer.9") or \
        name.startswith("encoder.layer.10") or \
        name.startswith("encoder.layer.11") or \
        name.startswith("pooler"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print(f"Trainable parameters: {sum(p.numel() for p in custom_model.parameters() if p.requires_grad)}")
    return custom_model

def define_optimizer(custom_model):
    optimizer = optim.AdamW(
    [
        {
            "params": [p for n, p in custom_model.named_parameters()
                       if "bert" in n and p.requires_grad],
            "lr": 2e-5,
        },
        {
            "params": [p for n, p in custom_model.named_parameters()
                       if "bert" not in n and p.requires_grad],
            "lr": 1e-4,
        },
    ],
    weight_decay=0.01,
    )
    return optimizer

def define_training_arguments(output_dir="./results", num_train_epochs=5, per_device_train_batch_size=16, per_device_eval_batch_size=16, weight_decay=0.01, logging_dir="./logs", logging_steps=100):
    training_args = TrainingArguments(
        output_dir=output_dir,           # Directory for saving model checkpoints
        eval_strategy="epoch",     # Evaluate at the end of each epoch
        save_strategy="epoch",
        learning_rate=5e-3,              # Start with a small learning rate
        per_device_train_batch_size=per_device_train_batch_size,  # Batch size per GPU
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,              # Number of epochs
        weight_decay=weight_decay,               # Regularization
        save_total_limit=2,              # Limit checkpoints to save space
        load_best_model_at_end=True,     # Automatically load the best checkpoint
        logging_dir=logging_dir,            # Directory for logs
        logging_steps=logging_steps,
        metric_for_best_model="f1_macro",
        report_to="none"
    )
    return training_args

def compute_metrics(eval_pred):
    f1_metric = load("f1")
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"f1_macro":f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"], "f1_weighted":f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]}

class ReviewsDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, metadata):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.metadata = metadata

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "metadata": self.metadata[idx]
        }
    
def create_datasets(tokenizer, train_input_ids, train_attention_mask, y_train, train_metadata, X_val, y_val, test_input_ids, test_attention_mask, y_test, test_metadata):
    train_dataset = ReviewsDataset(train_input_ids, train_attention_mask, y_train, train_metadata)
    val_tokenized = tokenizer(X_val['text'].tolist(), padding="max_length", truncation=True,max_length=128, return_tensors="pt")
    val_input_ids = val_tokenized['input_ids']
    val_attention_mask = val_tokenized['attention_mask']
    val_metadata = torch.from_numpy(X_val[['instructor_rating', 'recommendability', 'enjoyability', 'difficulty', 'hours_per_week']].values)
    val_metadata = val_metadata.float()
    y_val = torch.from_numpy(np.array(y_val)).long()
    val_dataset = ReviewsDataset(val_input_ids, val_attention_mask, y_val, val_metadata)
    test_dataset = ReviewsDataset(test_input_ids, test_attention_mask, y_test, test_metadata)
    return train_dataset, val_dataset, test_dataset

def define_trainer(custom_model, train_dataset, val_dataset, training_args, compute_metrics, tokenizer, data_collator, optimizer):
    # Define the Trainer
    trainer = Trainer(
        model=custom_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )
    trainer.train()
    return trainer

def evaluate_trainer(trainer, test_dataset, y_test):
    # Get raw predictions on the test set and print it into a report + confusion matrix
    pred_output = trainer.predict(test_dataset)
    logits = pred_output.predictions
    y_pred = logits.argmax(axis=-1)

    y_true = y_test.numpy()

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["not hurtful", "hurtful but useful", "hurtful"]
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return y_pred, y_true

def test_with_detoxify(X_test):
    texts = X_test['text'].astype(str).tolist()
    results = Detoxify('unbiased').predict(texts)
    return results