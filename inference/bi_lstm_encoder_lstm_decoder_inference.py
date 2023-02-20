# This notebook includes code to build the inference model for model 2 
# (model with stacked Bi-LSTM encoder and LSTM decoder with Bahdanau Attention), 
# as well as to generate predicted title from a given article

# I have consulted and adapted code from the following sources:
# - A. Pai, “Text Summarization: Text Summarization Using Deep Learning”, 2020 Analytics Vidhya. [Online]. 
#     Available: https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/. 
#     [Accessed: 21-Apr-2021]. 

import numpy as np  
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model, load_model
import os
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR) # hide tensorflow warnings

# %%
import sys
sys.path.append('../')
from util.preprocess_text import preprocess_text

# %% [markdown]
# # Load Saved Model

# %%
model_name = 'bi_lstm_encoder_lstm_decoder'

# %%
model = load_model("models/{}".format(model_name))


# %%
latent_dim = 128

# %%
max_len_full_article = model.inputs[0].shape[1]
max_len_title = 25

# %%
encoder_inputs = model.input[0]
encoder_outputs, state_forward_h, state_forward_c, state_backward_h, state_backward_c = model.layers[5].output
state_h = Concatenate()([state_forward_h, state_backward_h])
state_c = Concatenate()([state_forward_c, state_backward_c])
encoder_states = [state_h, state_c]

# %%
decoder_inputs = model.input[1]
decoder_embedding_layer = model.layers[6]
decoder_lstm = model.layers[9]
attention_layer = model.layers[10]
decoder_dense = model.layers[12]
# %% [markdown]
# # Build Inference Model

# %%
# Create encoder to generate feature vector from input sequence
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# Create decoder 
# Initialize tensors that will hold the internal states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim * 2,), name="input_3")
decoder_state_input_c = Input(shape=(latent_dim * 2,), name="input_4")
decoder_hidden_state_input = Input(shape=(max_len_full_article, latent_dim * 2), name="input_5")

# Get the embeddings of the decoder sequence
decooder_embedding_inference= decoder_embedding_layer(decoder_inputs) 
# assign the internal states from the prev time step to the initial states of the decoder lstm
# for predicting the next word in the sequence
decoder_outputs_inference, decoder_output_h_inference, decoder_output_c_inference = decoder_lstm(decooder_embedding_inference, initial_state=[decoder_state_input_h, decoder_state_input_c])

# Create attention layer for inference
attention_output_inference, _ = attention_layer([decoder_hidden_state_input, decoder_outputs_inference])
decoder_concat_attention_inference = Concatenate(axis=-1, name='concat')([decoder_outputs_inference, attention_output_inference])

# Crate dense softmax layer to generate prob distribution over the target vocabulary
decoder_outputs_inference = decoder_dense(decoder_concat_attention_inference) 

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs_inference] + [decoder_output_h_inference, decoder_output_c_inference])

# %%
'''
Decode sequence by using greedy decoding
'''
def decode_sequence(input_seq): 
    # Encode the input as state vectors
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Initialize the target sequence with the start token
    target_seq[0, 0] = target_word_index['sostoken']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
      
        output_tokens, output_h, output_c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Choose predicted token greedy
        predict_token_index = np.argmax(output_tokens[0, -1, :])
        predict_token = reverse_target_word_index[str(predict_token_index)]

        
        if(predict_token!='eostoken'):
            decoded_sentence += ' '+predict_token

        # Stop condition is either hitting max length for title or found the end token eostoken.
        if (predict_token == 'eostoken' or len(decoded_sentence.split()) >= (max_len_title-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = predict_token_index

        # Update internal states
        e_h, e_c = output_h, output_c

    return decoded_sentence

# %%
def convert_sequence_to_title(input_sequence):
    ret_title=''
    for word_token in input_sequence:
        if ((word_token!=0 and word_token!=target_word_index['sostoken']) and word_token!=target_word_index['eostoken']):
            ret_title = ret_title + reverse_target_word_index[str(word_token)] + ' '
    return ret_title

def convert_sequence_to_article(input_sequence):
    ret_article=''
    for word_token in input_sequence:
        if(word_token!=0):
            ret_article = ret_article+reverse_source_word_index[str(word_token)]+' '
    return ret_article

# %%
path_to_data = "train_val_nd_array"
x_train = np.load("{}/{}/x_train.npy".format(path_to_data, model_name))
y_train = np.load("{}/{}/y_train.npy".format(path_to_data, model_name))
x_validate = np.load("{}/{}/x_validate.npy".format(path_to_data, model_name))
y_validate = np.load("{}/{}/y_validate.npy".format(path_to_data, model_name))

# %%
import json 

with open("word_idx_dict/{}/reverse_source_word_index.json".format(model_name)) as f:
  reverse_source_word_index = json.load(f)

with open("word_idx_dict/{}/reverse_target_word_index.json".format(model_name)) as f:
  reverse_target_word_index = json.load(f)

with open("word_idx_dict/{}/target_word_index.json".format(model_name)) as f:
  target_word_index = json.load(f)


# %%
source_word_index = {word: index for index, word in reverse_source_word_index.items()}

# %%
def generate_title(article_file_txt): 
    with open(article_file_txt) as f:
        article = f.read()

    article = preprocess_text(article, is_article=True)
    input_sequence = [source_word_index[word] for word in article.split(" ") if word in source_word_index.keys()]

    padded_input_sequence = pad_sequences([input_sequence], maxlen=max_len_full_article, padding='post')
    generated_title = decode_sequence(padded_input_sequence.reshape(1, max_len_full_article))
    
    print("Here is a suggested title for the given essay: ")
    print("")
    
    return generated_title

