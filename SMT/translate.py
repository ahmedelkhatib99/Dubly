import imp
import pathlib
import random
import json
from tensorflow import keras
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.translate import ibm1, AlignedSent, Alignment, PhraseTable, StackDecoder
from collections import defaultdict
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from IBM_Model1.IBM_Model1 import IBM
import re

class SMT:

    def __init__(self) -> None:
        ibm_ourversion = IBM(IBM.EXECUTE_MODE)
        ibm_ourversion.load()
        self.phrase_table = self.phrase_table_generation(ibm_ourversion)
        self.language_model = self.language_model_generation()
        self.stack_decoder1 = StackDecoder(self.phrase_table, self.language_model)

    def language_model_generation(self):
        english_file = open(os.path.join(os.path.dirname(__file__), "./english.json"),"r")
        english_text = english_file.read()
        english_list = json.loads(english_text)
        fdist = defaultdict(lambda: 1e-300)
        #fdist = nltk.FreqDist(w for sentence in english_list for w in sentence)
        for sentence in english_list:
            for word in sentence:
                fdist[word] += 1
        language_model = type('', (object,),{'probability_change':lambda self,context,phrase:np.log(fdist[phrase]),'probability':lambda self,phrase:np.log(fdist[phrase])})()
        return language_model

    def phrase_table_generation(self, ibm1_model):
        phrase_table = PhraseTable()
        translation_table = ibm1_model.getTranslationTable() #change to ibm1_model
        for english_word in tqdm(translation_table):
            for spanish_word in translation_table[english_word].keys():
                phrase_table.add((spanish_word,),(english_word,), np.log(translation_table[english_word][spanish_word]))
        return phrase_table

    def translate(self, input_sentence):
        end_punctuation = input_sentence[-1]
        input_sentence =  re.sub(r"([?.!,¿])", r"", input_sentence)
        print(input_sentence)
        tokens = list(word_tokenize(input_sentence))
        translated = self.stack_decoder1.translate([word.lower() for word in tokens])
        return ' '.join(translated)+end_punctuation
