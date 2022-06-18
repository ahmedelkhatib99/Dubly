import torch
from torch import load
import pandas as pd
import unicodedata
import re
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# CONFIGURATIONS
device = torch.device("cpu")    
encoder_checkpoint_path = os.path.join(os.path.dirname(__file__), '.\\models\\encoder-310.pt')
decoder_checkpoint_path = os.path.join(os.path.dirname(__file__), '.\\models\\decoder-310.pt')
BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 1024


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dimension, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.gru = nn.GRU(embedding_dimension, self.encoder_units)
        
    def forward(self, x, lengths, device):
        x = self.embedding(x) 
        x = pack_padded_sequence(x, lengths)
        self.hidden = self.initialize_hidden_state(device)
        output, self.hidden = self.gru(x, self.hidden)
        output, _ = pad_packed_sequence(output)
        return output, self.hidden

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_size, self.encoder_units)).to(device)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dimension, decoder_units, encoder_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_units = decoder_units
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, embedding_dimension)
        self.gru = nn.GRU(embedding_dimension + encoder_units, 
                          self.decoder_units,
                          batch_first=True)
        self.fc = nn.Linear(encoder_units, self.vocab_size)
        self.W1 = nn.Linear(encoder_units, self.decoder_units)
        self.W2 = nn.Linear(encoder_units, self.decoder_units)
        self.V = nn.Linear(encoder_units, 1)
    
    def forward(self, x, hidden, enc_output):
        # max length, batch size, enc_units --> batch size, max length, hidden_size
        enc_output = enc_output.permute(1,0,2) 
        
        # batch size, hidden size --> batch size, 1, hidden size
        hidden_with_time_axis = hidden.permute(1, 0, 2)
        
        # Bahdanaus's score: batch size, max_length, hidden_size
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
          
        attention_weights = torch.softmax(self.V(score), dim=1)
        
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        
        x = self.embedding(x)
        
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        
        output, state = self.gru(x)
        
        hidden_size = output.size(2)
        output =  output.view(-1, hidden_size)
        
        x = self.fc(output)
        return x, state, attention_weights
    
    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_size, self.decoder_units))

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w

class Indexer():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
        
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

def add_padding(x, max_len):
    x_padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: x_padded[:] = x[:max_len]
    else: x_padded[:len(x)] = x
    return x_padded

def max_length(tensor):
        return max(len(t) for t in tensor)

def translate(sentence_to_translate:str):
    sentence_to_translate = preprocess(sentence_to_translate)
    f = open(os.path.join(os.path.dirname(__file__), '.\\dataset\\spa.txt'), encoding='UTF-8').read().strip().split('\n')  
    dataset_lines=f
    pairs_count = 30000 

    original_word_pairs = [[w for w in l.split('\t')[0:2]] for l in (dataset_lines[1000:1000+(pairs_count//2)] + dataset_lines[-(pairs_count//2):])]
    data = pd.DataFrame(original_word_pairs, columns=["eng", "es"])

    data["eng"] = data.eng.apply(lambda w: preprocess(w))
    data["es"] = data.es.apply(lambda w: preprocess(w))        

    spanish_indexer = Indexer(data["es"].values.tolist())
    english_indexer = Indexer(data["eng"].values.tolist())
    spanish_tensor = [[spanish_indexer.word2idx[s] for s in es.split(' ')]  for es in data["es"].values.tolist()]

    max_length_spanish = max_length(spanish_tensor)
    spanish_vocab_size = len(spanish_indexer.word2idx)
    english_vocab_size = len(english_indexer.word2idx)
    
    encoder = Encoder(spanish_vocab_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(english_vocab_size, EMBEDDING_DIM, UNITS, UNITS, BATCH_SIZE)

    encoder_checkpoint = load(encoder_checkpoint_path, map_location=device)
    encoder.load_state_dict(encoder_checkpoint["model_state"])
    decoder_checkpoint = load(decoder_checkpoint_path, map_location=device)
    decoder.load_state_dict(decoder_checkpoint["model_state"])
    encoder.to(device)
    decoder.to(device)

    with torch.no_grad():
        spanish_sequence =[spanish_indexer.word2idx[s] for s in sentence_to_translate.split(' ')]
        spanish_tensor = [add_padding(spanish_sequence, max_length_spanish)] * 64 
        lengths = [ np.sum(1 - np.equal(x, 0)) for x in spanish_tensor]
        
        encoder.eval()
        decoder.eval()
        encoder.initialize_hidden_state(device)
        encoder_output, encoder_hidden = encoder(torch.tensor(spanish_tensor).transpose(0,1).to(device), torch.tensor(lengths), device)

        decoder.initialize_hidden_state()
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[english_indexer.word2idx['<start>']]] * BATCH_SIZE)
        # print("--------")

        output = []
        for i in range(len(spanish_sequence)):
            predictions, decoder_hidden, _ = decoder(decoder_input.to(device), 
                                                decoder_hidden.to(device), 
                                                encoder_output.to(device))
            _, token_ids = predictions.data.topk(1)
            token_id = int(token_ids[0][0])
            if token_id == english_indexer.word2idx['<end>']:
                break
            output.append(english_indexer.idx2word[token_id])
            # print("Decoder Hidden: ", predictions.shape)
            decoder_input = torch.tensor([[token_id]] * BATCH_SIZE)
        print("Prediction: ", ' '.join(output))
        return ' '.join(output)