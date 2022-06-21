import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from sklearn.metrics.pairwise import cosine_similarity
from model import decode_sequence
from model import eng_vectorization

def calculate_accuracy(val_pairs, max_length):
  c = 0
  sum = 0
  for pair in val_pairs[:max_length]:
    translated = decode_sequence(pair[0])
    similarity = cosine_similarity(eng_vectorization([pair[1]]), eng_vectorization([translated]))
    sum += similarity
  return sum/max_length