
import string
import pathlib
import json
from tensorflow import keras
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import math
# nltk.download('punkt')
import os

class IBM:
  MAX_NUM_OF_ITERATIONS = 10
  MIN_PROB = 1.0e-12
  EPSILON = 1e-20
  ENGLISH_SENTENCE_INDEX = 0
  SPANISH_SENTENCE_INDEX = 1
  TRAINIG_MODE = True
  EXECUTE_MODE = False
  def __init__(self, isTrain = False) -> None:
    if isTrain:
      text_file = keras.utils.get_file(
      fname="spa-eng.zip",
      origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
      extract=True,
      )  
      self.text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"
    self.t_spa_en_mat = None


  def build_languages_vocabulary(self, sentences_dataset):
        spanish_vocab = set()
        english_vocab = set()
        for aligned_sentence in sentences_dataset:
            english_vocab.update(aligned_sentence[IBM.ENGLISH_SENTENCE_INDEX])
            spanish_vocab.update(aligned_sentence[IBM.SPANISH_SENTENCE_INDEX])
        # Add the NULL token
        spanish_vocab.add(None)
        self.spanish_vocab = spanish_vocab
        self.english_vocab = english_vocab


  def init_probabilities(self):
        self.translation_table = defaultdict(
            lambda: defaultdict(lambda: IBM.MIN_PROB)
        )
        self.alignment_table = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: IBM.MIN_PROB))
            )
        )
  
  def train(self):
    en_sentences, spa_sentences  = self.preprocess() #2 arrays of english and spanish sentences
    en_tokenized_sentences = self.tokenize_sentences(en_sentences) #array of english tokens [['a','b'],[],...]
    spa_tokenized_sentences = self.tokenize_sentences(spa_sentences) #array of spanish tokens
    bilingual_tokenized_text = []
    for iter in zip(en_tokenized_sentences, spa_tokenized_sentences):
        bilingual_tokenized_text.append([iter[IBM.ENGLISH_SENTENCE_INDEX],iter[IBM.SPANISH_SENTENCE_INDEX]]) #[ [entokens1,sptokens1], [en2,sp2], [en3,sp3], ... ]
    
    self.build_languages_vocabulary(bilingual_tokenized_text)
    self.init_probabilities()

    self.expectation_maximization(bilingual_tokenized_text)

    out_file = open("translation_table.json", "w")
    json.dump(self.translation_table, out_file, indent = 6)
    out_file.close()
    

  def is_converged(self, num_of_iterations):
    if num_of_iterations > self.MAX_NUM_OF_ITERATIONS :
        return True
    return False

  def preprocess(self):
    with open(self.text_file, encoding="utf8") as f:
      lines = f.read().split("\n")[:-1]
    en_sentences = []
    spa_sentences = []
    for line in lines:
        eng, spa = line.split("\t")
        en_sentences.append(eng)
        spa_sentences.append(spa)
    return en_sentences, spa_sentences

  def tokenize_sentences(self, sentence_list):
    tokenized_sentences = []
    translate_table = dict((ord(char), None) for char in string.punctuation)
    for sentence in sentence_list:
        sentence = sentence.translate(translate_table)
        tokens = word_tokenize(sentence.lower())
        tokenized_sentences.append(tokens)
    return tokenized_sentences


  def expectation_maximization(self, sentences_dataset):
    count_en_spa = defaultdict(lambda: defaultdict(lambda: 0.0))
    total_spa = defaultdict(lambda: 0.0)
    initial_prob = 1 / len(self.english_vocab)
    for t in self.english_vocab:
        self.translation_table[t] = defaultdict(lambda: initial_prob)

    num_of_iter = 0
    while not self.is_converged(num_of_iter):
      num_of_iter += 1

      # Initialization of variables
      for aligned_sentences in sentences_dataset:
        english_sentence = aligned_sentences[IBM.ENGLISH_SENTENCE_INDEX]
        spanish_sentence = aligned_sentences[IBM.SPANISH_SENTENCE_INDEX]
        e_total = defaultdict(lambda: 0.0)
        for en_word in english_sentence:
            for spa_word in spanish_sentence:
                e_total[en_word] += self.translation_table[en_word][spa_word] #sum of probabilities between en1 and spa (probability of english word)
        
        # collect counts
        for en_word in english_sentence:
          for spa_word in spanish_sentence:
              count_en_spa[en_word][spa_word] += self.translation_table[en_word][spa_word] / e_total[en_word] #p(e|s) = p(s inter e)/p(e)
              total_spa[spa_word] += self.translation_table[en_word][spa_word] / e_total[en_word]


      # estimate probabilities
      for en_word, spa_words in count_en_spa.items():
          for spa_word in spa_words:
            if count_en_spa[en_word][spa_word] != 0:
                self.translation_table[en_word][spa_word] = count_en_spa[en_word][spa_word] / total_spa[spa_word]

    # end while

  def load(self):
    # Load the language maps
    with open(os.path.join(os.path.dirname(__file__), "../translation_table.json"), "r") as f:
      self.translation_table = json.load(f)

  def getTranslationTable(self):
    return self.translation_table

  def translate(self, en_word, spa_word):
    return self.translation_table[en_word][spa_word]

  def test(self):
    print((self.t_spa_en_mat[self.t_spa_en_mat == np.min(self.t_spa_en_mat)]).shape, self.t_spa_en_mat.shape)

# IBM(IBM.TRAINIG_MODE).train()