import os
import pickle
import string
import re
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from keras.layers import TextVectorization

def eng_custom_standardization(input_string):
        eng_strip_chars = string.punctuation.replace("[", "")
        eng_strip_chars = eng_strip_chars.replace("]", "")
        lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(eng_strip_chars), "")

def spa_custom_standardization(input_string):
    spa_strip_chars = string.punctuation + "¿"
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(spa_strip_chars), "")

def translate(spa_texts, eng_custom_standardization, spa_custom_standardization):
    
    vocab_size = 15000
    sequence_length = 25
    batch_size = 64

    eng_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=eng_custom_standardization,
    )
    spa_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize=spa_custom_standardization,
    )
    ##############################################

    #Loading transformer and vectorization of both languages
    from_disk = pickle.load(open(os.path.dirname(__file__) + "\\spa_vectorization.pkl", "rb"))
    spa_vectorization = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    spa_vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    spa_vectorization.set_weights(from_disk['weights'])


    from_disk = pickle.load(open(os.path.dirname(__file__) + "\\eng_vectorization.pkl", "rb"))
    eng_vectorization = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    eng_vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    eng_vectorization.set_weights(from_disk['weights'])


    transformer = tf.saved_model.load(os.path.dirname(__file__) + '\\translator-transformer')
    ##############################################

    eng_vocab = eng_vectorization.get_vocabulary()
    eng_index_lookup = dict(zip(range(len(eng_vocab)), eng_vocab))
    max_decoded_sentence_length = 25

    #decode_sequence
    eng_texts = []
    for input_sentence in spa_texts:
        tokenized_input_sentence = spa_vectorization([input_sentence])
        decoded_sentence = "[start]"
        for i in range(max_decoded_sentence_length):
            tokenized_target_sentence = eng_vectorization([decoded_sentence])[:, :-1]
            predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = eng_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == "[end]":
                break
        decoded_sentence = decoded_sentence.replace("[start] ", "")
        decoded_sentence = decoded_sentence.replace(" [end]", "")
        decoded_sentence = decoded_sentence + "."
        eng_texts.append(decoded_sentence)
        
    return ' '.join(eng_texts)