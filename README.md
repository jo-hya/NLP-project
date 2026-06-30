# NLP-project

NOTE: This project was developed and evaluated inside Google Colab to utilize cloud GPU resources. The final production-ready scripts and notebooks have been consolidated here for version control and portfolio presentation. Thus, everything was committed to the main branch directly.

An NLP-based project built to classify user-generated course reviews for theCourseForum at UVA and help solve the issue of misclassification of reviews with context-dependent language. For example, a review may use profanity or words with a negative connotation in a positive context, leading to unnecessary. Thus, we use a BERT-based model for context-aware sentiment analysis and moderation of course reviews to distinguish between harmful or inappropriate reviews from constructive but strongle worded ones. The goal is to help theCourseForum balance safety for UVA students and professors while supporting open and authentic feedback.

## NLP Pipeline

* Collected course reviews from theCourseForum and manually label data
* Preprocessed data and cleaned data to remove typos and improper formatting, and split the data into train, validation, and test data sets
* Balance an imbalanced training dataset by oversampling minority classes
* Created embeddings with concatenated text tokenization and numerical metadata
* Created a custom BERT model and added class weights to fix class imbalance
* Trained BERT model
* Evaluated the model

## Technologies Used

* Python
* PyTorch
* Scikit-learn
* Numpy
* Pandas
* Transformers

## Repository Structure

* tCF Review Data - reviews_last_8y.csv: manually labeled reviews from theCourseForum
* visualize.py: analyzes several aspects of the data such as the frequency of different words in the reviews, the number of reviews per semester, and the number of reviews per department
* .png files: the resulting bar charts from running visualize.py
* nlp_model.py: the full end-to-end NLP pipeline from the loading and preprocessing of review data all the way to the evaluation of the model
* main.py: high-level overview of the entire NLP pipeline and this is the file to run

## How to Run the Code

`python main.py` to run the model OR
`python visualize.py` to generate the bar charts