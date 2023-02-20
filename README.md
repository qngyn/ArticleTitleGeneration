# Generate Articles' Titles with Context using Recurrent Neural Network

A collection of Recurrent Neural Network (RNN) seq-2-seq models with Bi-LSTM / LSTM encoder and LSTM decoder using attention to generate titles from articles with context

- user_interact.py: 
    - Python program for user to try out the models through command-line interface
    - Run with "python3 user_interact.py"
    - May take a while to load (~ 1min for my machine)

* NOTE:
I am using `pyinquirer` lib version 1.0.3, which requires `prompt-toolkit==1.0.14`
However, if you run .ipynb file, it will need (and it would automatically install) a higher version of prompt-toolkit (3.0.18 for my machine)
(as shown in this issue here on the library's GitHub https://github.com/CITGuru/PyInquirer/issues/1#issuecomment-627524817)
So if you happen to run the .ipynb, or if the error "cannot import name 'Token' from 'prompt_toolkit.token'" ever comes up
please run "pip3 install prompt_toolkit==1.0.14" before running user_interact.py. 

- eval_pred.ipynb: including code to generate the ROUGE-1 and ROUGE-L F-similarity score for the models' predictions on 1000 samples in the validate set

- ./sample_essays:
    - Folder including .txt files containing some example essays that can be used to try out the model (e.g. through user_interact.py)

- ./inference: 
    - Folder containing code for inference models

- ./build_model:
    - .ipynb files: Jupyter Notebook containing code to build and train models
        + These notebooks were run on Google Colaboratory as well as Vassar's lambda-quad machine
        so the model names and file paths included are different, thus they may not compile successfully locally here
    - articles_2.json: JSON file containing 100,000 articles used to train the models
    - glove.6B.200d.txt: GloVe embeddings
    - attention.py: custom Bahdanau Attention layer in Keras
        + Written by Thushan Ganegedara (https://github.com/thushv89/attention_keras)
        + I did not alter this file

- ./models: 
    - Folder containing the saved models
    - Not included here due to large size

- ./training_val_nd_array: 
    - Tokenized training and validate sets of word sequence for each model
    - They should be the same for all of the models, only separated for consistency in naming file paths

- ./word_idx_dict: 
    - Folder containing dictonaries in the form of JSON of words in the corpus of each model and their corresponding unique token id

- ./training_history: 
    - folder containing the training results of the models in the form of csvs

- ./util:
    - preprocess_text.py: python program to preprocess the input essay/article

- requirements.txt: 
    - List of dependencies to install before running user_interact.py

Code reference:
- A. Pai, “Text Summarization: Text Summarization Using Deep Learning”, 2020 Analytics Vidhya. [Online]. Available: https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/. [Accessed: 21-Apr-2021]. 
- F. Chollet, “Keras documentation: Using pre-trained word embeddings,” Keras, 2020, [Online]. Available: https://keras.io/examples/nlp/pretrained_word_embeddings/. [Accessed: 27-May-2021]. 
- “Neural machine translation with attention,” TensorFlow. [Online]. Available: https://www.tensorflow.org/tutorials/text/nmt_with_attention. [Accessed: 27-May-2021]. 
- T. Ganegedara, “Keras Attention Layer,” GitHub, 2020. [Online]. Available: https://github.com/thushv89/attention\_keras. [Accessed: 27-May-2021]. 
