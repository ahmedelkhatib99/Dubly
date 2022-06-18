import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from sklearn.metrics.pairwise import cosine_similarity
from translate import translate
from translate import add_padding


def calculate_accuracy(loaded_encoder, loaded_decoder, inp_lang_validation, targ_lang_validation, targ_lang, max_length_inp):
  c = 0
  sum = 0
  for pair in list(zip(inp_lang_validation, targ_lang_validation))[:30]:
    translated = '<start> ' + translate(loaded_encoder, loaded_decoder, pair[0]) + ' <end>'
    translated_vector = add_padding([targ_lang.word2idx[s] for s in pair[1].split(' ')], max_length_inp)
    original_vector = add_padding([targ_lang.word2idx[s] for s in translated.split(' ')], max_length_inp)
    similarity = cosine_similarity([translated_vector], [original_vector])[0][0]
    print("Original   : ", pair[1])
    print("Translation: ", translated)
    print("Cosine similarity:", similarity)
    print()
    c += 1
    sum += similarity
  return sum/c

#print("avg cosine similarity: ", calculate_accuracy(loaded_encoder, loaded_decoder, inp_lang_validation, targ_lang_validation, targ_lang, max_length_inp))