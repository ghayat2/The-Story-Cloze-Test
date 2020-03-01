# The-Story-Cloze-Test
Given a four-sentence short story and two alternatives for the last, fifth sentence that ends the story, we built a system that aimed to decide which of the two is correct.


# Story Cloze Test - NLU Project 2
The goal of this project is to predict the ending of a story. To see the detailed task as well as our approach, please refer to the report. 

## Data
The following datasets are required:
- training set: containing 88161 five-sentence short stories that include only the correct ending
- validation set: containing 1871 stories with positive and negative endings
- test set: containing stories with two endings
- 100 dimension word2vec embedding
- Google 300 dimension word2vec embeddings
- 300 dimension Glove embeddings
- Pretrained skipthought embeddings.

All files should be put in a `data/` file, they are available for download [here](https://polybox.ethz.ch/index.php/apps/files/?dir=/Shared/data&fileid=1404350091)

## Setup
- Run the command:
```bash
 pip3 install --user -r requirements.txt
```
- Create a folder "data/processed/"
- Run the following commands
```bash
python3 process_train.py
python3 process_eval.py eval_stories.csv data/tokenizer.pickle
python3 process_test.py test-stories.csv data/tokenizer.pickle
python3 process_test.py test_for_report-stories_labels.csv data/tokenizer.pickle
```

These command should create a vocabulary and encode the training/evaluation set for the given task.

To train the model, run the command:
```bash
python3 train.py
```

The relevant flags are the following:
- use_train_set: Whether to use train set or eval set for training (default True)
- word_embedding_dimension: Word embedding dimension size (default: 100, google and glove should be 300)
- ratio_neg_random: Ratio of negative random endings (default: 4)
- ratio_neg_back: Ratio of negative backward endings (default: 2)
- embeddings: embeddings to use (default: w2v, options: w2v, w2v_google, glove)
- use_skip_thoughts: Whether we use skip thoughts for sentences embedding (default: true)
- use_pronoun_contrast: Whether the pronoun contrast feature vector should be added to the networks' input (default true)
- use_n_grams_overlap: Whether the n grams overlap feature vector should be added to the network's input.(default true)
- use_sentiment_analysis: Whether to use the sentiment intensity analysis (4 dimensional vectors (default true)
- attention: Whether to use attention (default: none, options: add ~ Bahdanau, mult ~ Luong)
- batch_size: (default 16)

## Assessing Performance
After having pre-processed the testing files using the commands above,
you can test our runs using `evaluate.py`. Here is a summary of the arguments
and flags that you will need to pay attention to using this script:
- The first and only argument is the name of a directory in `/runs`
which contains the checkpoints of a trained model.
```bash
python3 evaluate.py checkpoint_file
```
- Tensorflow glags:
    - predict: Boolean indicating if you want to make predictions on the
dataset "test_for_report-stories_labels.csv" or rather use "test-stories.csv"
and calculate the average accuracy of the model.
    - used_features: Boolean indicating if any feature extraction method
    was used during training. If any of the three flags below is true,
    then this flag must be set to true as well. 
    - use_pronoun_contrast: If the pronoun contrast feature extraction method
    was used during training.
    - use_n_grams_overlap: If the ngrams overlap feature extraction method
    was used during training. 
    - use_sentiment_analysis=If the sentiment analysis feature extraction method
    (Vader) was used during training. 
    - attention: Either "add", "mult" or `None` to indicate if an attention
    method was used and if so, which one between Bahdanau (add) or 
    Luong (mult).
    - embeddings: embeddings to use (default: w2v, options: w2v, w2v_google, glove)
    - word_embedding_dimension: Word embedding dimension size (default: 100, google and glove should be 300)
    - use_skip_thoughts: Whether we use skip thoughts for sentences embedding (default: true)
    - use_validation_set: calculate the average accuracy on the validation set (default:false)
    

