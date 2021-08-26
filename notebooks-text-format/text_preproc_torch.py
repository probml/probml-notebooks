# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: 'Python 3.6.7 64-bit (''base'': conda)'
#     name: python367jvsc74a57bd050da0f6fa72fb86d21724871d314354b884db45bd357078f1680189ca335f685
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/text_preproc_torch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="Yn51eYujm5S1"
# # Text preprocessing
#
# We discuss how to convert a sequence of words or characters into numeric form, which can then be fed into an ML model.
#
#
#

# + id="ysx0t0REm4r0"
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=1)
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

# !mkdir figures # for saving plots


# + id="V6Jbluorndzr"
import collections
import re
import random
import os
import requests
import zipfile
import hashlib


# +
# Required functions for downloading data

def download(name, cache_dir=os.path.join('..', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


# + [markdown] id="e9vbpUMwTRY1"
# # Basics
#
# This section is based on sec 8.2 of http://d2l.ai/chapter_recurrent-neural-networks/text-preprocessing.html
#

# + [markdown] id="RMrGxkRNnOx_"
# ## Data
#
# As a simple example, we use the book "The Time Machine" by H G Wells, since it is short (30k words) and public domain.

# + colab={"base_uri": "https://localhost:8080/"} id="D7OJT7o8nDQN" outputId="2dd7e687-6b9a-48d2-ab49-e1bb5b64d41b"
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  
    """Load the time machine dataset into a list of text lines."""
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'number of lines: {len(lines)}')


# + colab={"base_uri": "https://localhost:8080/"} id="uCsuaurvnlK8" outputId="8dcf484e-8f45-4748-cc93-c2a2ada427d2"
for i in range(11):
  print(i, lines[i])

# + colab={"base_uri": "https://localhost:8080/"} id="btVyl4dItGVT" outputId="6b67aec4-4c26-43f7-ea6e-0c7fd1440f55"
nchars = 0
nwords = 0
for i in range(len(lines)):
  nchars += len(lines[i])
  words = lines[i].split()
  nwords += len(words)
print('total num characters ', nchars)
print('total num words ', nwords)


# + [markdown] id="KKBbwDcKnwsA"
# ## Tokenization

# + colab={"base_uri": "https://localhost:8080/"} id="X32lM-XvnxhC" outputId="4783ff38-b282-4b0e-fc59-996f6ec0d6a6"
def tokenize(lines, token='word'):  
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


# + [markdown] id="X-Tbg9jjn8XN"
# ## Vocabulary
#
# We map each word to a unique integer id, sorted by decreasing frequency.
# We reserve the special id of 0 for the "unknown word".
# We also allow for a list of reserved tokens, such as “pad" for padding, "bos" to present the beginning for a sequence, and “eos” for the end of a sequence.
#

# + id="8ZOLrVNon9dk"
class Vocab:  
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The index for the unknown token is 0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(tokens):  
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# + [markdown] id="CV0rTlaqoSNE"
# Here are the top 10 words (and their codes) in our corpus.

# + colab={"base_uri": "https://localhost:8080/"} id="tYmbCwY6oUFB" outputId="31a05a85-5113-4db8-aacf-f944f2c576f8"
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

# + [markdown] id="sKXJQdbXoiqT"
# Here is a tokenization of a few sentences.

# + colab={"base_uri": "https://localhost:8080/"} id="jd73-1zzoUWo" outputId="f2e7dbda-4053-4773-d385-686f6c549144"
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])


# + [markdown] id="-6LsXchMop3u"
# ## Putting it altogether
#
# We tokenize the corpus at the character level, and return the sequence of integers, as well as the corresponding Vocab object.

# + id="1BywQ9iUoq_D"
def load_corpus_time_machine(max_tokens=-1): 
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab



# + colab={"base_uri": "https://localhost:8080/"} id="oQzX4Am8osdh" outputId="53318718-5dba-4574-8f7d-3de538584c0d"
corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)

# + colab={"base_uri": "https://localhost:8080/"} id="IgDxt_PRovAb" outputId="7aaffeaa-0b08-4796-bce7-e3ef6fe9cc58"
print(corpus[:20])

# + colab={"base_uri": "https://localhost:8080/"} id="egONc6CRowLa" outputId="030047c5-7199-4000-9e0c-39ba75275fd6"
print(list(vocab.token_to_idx.items())[:10])


# + colab={"base_uri": "https://localhost:8080/"} id="9xKwPjAAozaX" outputId="68c7959c-7bfe-42a1-f324-b9b4a1a90113"
print([vocab.idx_to_token[i] for i in corpus[:20]])

# + [markdown] id="X3fLUodCZebY"
# ## One-hot encodings
#
# We can convert a sequence of N integers into a N*V one-hot matrix, where V is the vocabulary size.

# + colab={"base_uri": "https://localhost:8080/"} id="Qk21iCFhZj89" outputId="b5323706-52f8-459b-b382-70acbf54ba26"
x = torch.tensor(corpus[:3])
print(x)
X = F.one_hot(x, len(vocab))
print(X.shape)
print(X)


# + [markdown] id="8vO99OOSuYhX"
# # Language modeling
#
# When fitting language models, we often need to chop up a long sequence into a set of short sequences, which may be overlapping, as shown below, where we extract subsequences of length $n=5$. 
#
# <img src="https://github.com/probml/pyprobml/blob/master/images/timemachine-5gram.png?raw=true">
#
# Below we show how to do this.
#
# This section is based on sec 8.3.4 of
# http://d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html#reading-long-sequence-data
#

# + [markdown] id="Vert2-4qw5K7"
# ## Random ordering

# + [markdown] id="_rARqDyZuvlu"
# To increase variety of the data, we can start the extraction at a random offset. We can thus create a random sequence data iterator, as follows.
#

# + id="meuw3vkjpL22"
def seq_data_iter_random(corpus, batch_size, num_steps):  
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


# + [markdown] id="71kdus7mvMFQ"
# For example, let us generate a sequence 0,1,..,34, and then extract subsequences of length 5. Each minibatch will have 2 such subsequences, starting at random offsets. There is no ordering between the subsequences, either within or across minibatches. There are $\lfloor (35-1)/5 \rfloor = 6$ such subsequences, so the iterator will generate 3 minibatches, each of size 2.
#
# For language modeling tasks, we define $X$ to be the first $n-1$ tokens, and $Y$ to be the $n$'th token, which is the one to be predicted.

# + colab={"base_uri": "https://localhost:8080/"} id="x8GXyqOgvOI7" outputId="efc1667a-624e-461c-a72c-c9d8ac244f81"
my_seq = list(range(35))
b = 0
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('batch: ', b)
    print('X: ', X, '\nY:', Y)
    b += 1


# + [markdown] id="wdg490Gow7la"
# ## Sequential ordering

# + [markdown] id="55ECVkQLwL8K"
# We can also require that the $i$'th subsequence in minibatch $b$ follows the $i$'th subsequence in minibatch $b-1$. This is useful when training RNNs, since when the model encounters batch $b$, the hidden state of the model will already be initialized by the last token in sequence $i$ of batch $b-1$.

# + id="r3uVV7lYwCdv"
def seq_data_iter_sequential(corpus, batch_size, num_steps):  
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


# + [markdown] id="KGRIkFvXwZX6"
# Below we give an example. We see that the first subsequence in batch 1
# is [0,1,2,3,4], and the first subsequence in batch 2 is [5,6,7,8,9], as desired.

# + colab={"base_uri": "https://localhost:8080/"} id="aLQzm2qrwY0m" outputId="27a0f0e7-46db-469c-dd1f-906b599be01d"
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


# + [markdown] id="SP96EBA-w9MF"
# ## Data iterator
# -

def load_corpus_time_machine(max_tokens=-1):
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# + id="IpjIv8tMw-QD"
class SeqDataLoader:  #@save
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


# + id="pIy_YUk9w-0A"
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter,
                              max_tokens)
    return data_iter, data_iter.vocab


# + id="Y44o8-MkxA3t"
data_iter, vocab = load_data_time_machine(2, 5)


# + colab={"base_uri": "https://localhost:8080/"} id="sf-py0roxmAC" outputId="66691d42-4905-4fe3-b906-8856e525665b"
print(list(vocab.token_to_idx.items())[:10])

# + colab={"base_uri": "https://localhost:8080/"} id="6XhwWfMHxXTA" outputId="a29066f2-d2e2-447c-d290-ea473ecc0ee1"
b = 0
for X, Y in data_iter:
    print('batch: ', b)
    print('X: ', X, '\nY:', Y)
    b += 1
    if b > 2:
      break

# + [markdown] id="yDmK1xQ9T4IY"
# # Machine translation
#
# When dealing with sequence-to-sequence tasks, such as NMT, we need to create a vocabulary for the source and target language. In addition, the input and output sequences may have different lengths, so we need to use padding to ensure that we can create fixed-size minibatches. We show how to do this below.
#
# This is based on sec 9.5 of 
# http://d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html
#
#
#

# + [markdown] id="gBUgcAcmUdCJ"
# ## Data
#
# We use an English-French dataset that consists of bilingual sentence pairs from the [Tatoeba Project](http://www.manythings.org/anki/). Each line in the dataset is a tab-delimited pair of an English text sequence (source) and the translated French text sequence (target).
#

# + colab={"base_uri": "https://localhost:8080/"} id="UnjXAtdYUUW8" outputId="66c19d66-217b-43ef-9877-854ea32725d0"
DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    """Load the English-French dataset."""
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:100])


# + [markdown] id="sqZImjzDVMHa"
# ## Preprocessing
#
# We apply several preprocessing steps: we replace non-breaking space with space, convert uppercase letters to lowercase ones, and insert space between words and punctuation marks.
#

# + colab={"base_uri": "https://localhost:8080/"} id="r5ZUH4ZaUquY" outputId="316a660f-1d3a-45e4-ddb2-f97ed3128733"
def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [
        ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:110])


# + [markdown] id="yGChAGPjVgUn"
# We tokenize at the word level.  The following tokenize_nmt function tokenizes the the first `num_examples` text sequence pairs, where each token is either a word or a punctuation mark. 

# + id="PZ-iR79zVKM_"
def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target



# + colab={"base_uri": "https://localhost:8080/"} id="v282HIgRVfza" outputId="12cafea1-eee6-43e9-ba8d-79c6ef5affc2"
source, target = tokenize_nmt(text)
source[:10], target[:10]

# + [markdown] id="LC8u2YndV6-P"
# ## Vocabulary
#
# We can make a source and target vocabulary. To avoid having too many unique tokens, we specify a minimum frequency of 2 - all others will get replaced by "unk". We also add special tags for padding, begin of sentence, and end of sentence.

# + colab={"base_uri": "https://localhost:8080/"} id="v9SrJQc3VtSn" outputId="8b0e6cd5-890d-46c5-ce79-14f256ba63b3"
src_vocab = Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)

# + colab={"base_uri": "https://localhost:8080/"} id="KPF2FthfV840" outputId="f08bca8e-8564-411d-e58f-6e5b2e19f671"
# French has more high frequency words than English
target_vocab = Vocab(target, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(target_vocab)


# + [markdown] id="d0DyArAtWcob"
# ## Truncation and padding
#
# To create minibatches of sequences, all of the same length, we truncate sentences that are too long, and pad ones that are too short.

# + colab={"base_uri": "https://localhost:8080/"} id="x2D62tczWP-2" outputId="25ca1de2-97a6-4273-abbf-644a41562bb8"
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

print(truncate_pad(source[0], 10, 'pad'))
print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))


# + id="RgyPxL6tWvJC"
def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([
        truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# + colab={"base_uri": "https://localhost:8080/"} id="SlwULfBwW9ma" outputId="51eae96b-bfd0-4325-93ef-c98315abb6b0"
num_steps = 10
src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
print(src_array.shape)
print(src_valid_len.shape)

# + colab={"base_uri": "https://localhost:8080/"} id="br6L3nDbXFHY" outputId="e70924a1-6e71-49dc-b393-364feac92296"
print(src_array[0,:]) # go, ., eos, pad, ..., pad
print(src_valid_len[0])


# + [markdown] id="UyXmgFUvXVnA"
# ## Data iterator
#
# Below we combine all of the above pieces into a handy function.

# + id="AkD1QMiJXKAP"
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


# + [markdown] id="ARsiX21oXdOd"
# Show the first minibatch.

# + colab={"base_uri": "https://localhost:8080/"} id="vl00eydyXeYF" outputId="a48df679-efd7-4e87-b3cd-fb66689e56e0"
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('valid lengths for Y:', Y_valid_len)
    break

# + id="thnQxtaIXenj"

