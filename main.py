import nlp_model

df = nlp_model.read_data()
X_train, y_train, X_val, y_val, X_test, y_test = nlp_model.preprocess_data(df)
tokenizer, train_input_ids, train_attention_mask, train_metadata, y_train, test_input_ids, test_attention_mask, test_metadata, y_test, data_collator = nlp_model.tokenize_data(X_train, X_test)
custom_model = nlp_model.create_custom_model(train_metadata, y_train)
custom_model = nlp_model.freeze_bert_layers(custom_model)
optimizer = nlp_model.define_optimizer(custom_model)
training_args = nlp_model.define_training_arguments()
train_dataset, val_dataset, test_dataset = nlp_model.create_datasets(tokenizer, train_input_ids, train_attention_mask, y_train, train_metadata, X_val, y_val, test_input_ids, test_attention_mask, y_test, test_metadata)
trainer = nlp_model.define_trainer(custom_model, train_dataset, val_dataset, training_args, tokenizer, data_collator, optimizer)
y_pred, y_true = nlp_model.train_and_evaluate(trainer, test_dataset)

# compare results with detoxify api

detoxify_results = nlp_model.test_with_detoxify(X_test)