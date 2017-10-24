""" 
Pre-processing for the Cornell Movie-Dialogs Corpus.
"""
from __future__ import print_function

import os
import random
import re

import numpy as np

import config

############# Process raw text files

def getLineId2LineTextDictionary():
    """
    Parse movie_lines.txt and return dictionary of lineId to lineText
    """
    id2line = {}
    file_path = os.path.join(config.DATA_PATH, 'movie_lines.txt')
    with open(file_path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line

def getConversationsList():
    """ Get list of conversations from movie_conversations.txt """
    file_path = os.path.join(config.DATA_PATH, 'movie_conversations.txt')
    conversationsList = []
    with open(file_path, 'rb') as f:
        for line in f.readlines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(', '):
                    convo.append(line[1:-1])
                conversationsList.append(convo)

    return conversationsList

###########   

def conversationToQuestionAnswerPairs(id2line, conversationsList):
    """ 
    Divide the dataset into two sets: questions and answers. 
    Take the first line of the conversation as question and second line as answer, and vice versa!!!  [A,B,C] becomes (A,B), (B,C)
    """
    questions, answers = [], []
    i = 0
    for convo in conversationsList:
        for index, line in enumerate(convo[:-1]):
            questions.append(id2line[convo[index]])
            answers.append(id2line[convo[index + 1]])

    assert len(questions) == len(answers)
    return questions, answers

def createTrainTestEncoderDecoderDataSets(questions, answers):
    """ 
    Create train & test encoder & decoder files.
    enc / dec means encoder input / decoder output (question / answer)
    """
    make_dir(config.PROCESSED_PATH)
    
    # random conversationsList to create the test set
    test_ids = random.sample([i for i in range(len(questions))], config.TESTSET_SIZE)
    
    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []
    for filename in filenames:
        files.append(open(os.path.join(config.PROCESSED_PATH, filename),'wb'))

    for i in range(len(questions)):
        if i in test_ids:
            files[2].write(questions[i] + '\n')
            files[3].write(answers[i] + '\n')
        else:
            files[0].write(questions[i] + '\n')
            files[1].write(answers[i] + '\n')

    for file in files:
        file.close()

def make_dir(path):
    """ 
    Create a directory if there isn't one already. 
    """
    try:
        os.mkdir(path)
    except OSError:
        pass

def tokenize(line, normalize_digits=True):
    """ 
    Tokenize text into tokens.
    """
    # remove placeholders
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile(b"([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            # replace all digits to # character
            if normalize_digits:
                token = re.sub(_DIGIT_RE, b'#', token)
            words.append(token)
    return words

def build_vocab(filename, normalize_digits=True):
    """
    Build file of vocabularies that occur more than config.THRESHOLD
    in the dataset
    """
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))

    # Build dictionary of each vocabulary to its frequency 
    vocab = {}
    with open(in_path, 'rb') as f:
        for line in f.readlines():
            for token in tokenize(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    # Sort dictionary by count and get keys ordered by count
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'wb') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n') 
        index = 4
        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                break
            f.write(word + '\n')
            index += 1

def load_vocab(vocab_path):
    """
    Load vocabs file and 
    return list of all words and dictionary of each word to its index
    """
    with open(vocab_path, 'rb') as f:
        words = f.read().splitlines() 
    # get list of all words and dictionary of each word to its index
    return words, {words[i]: i for i in range(len(words))}

def stringToTokenIds(vocab, line):
    """
    Convert a sentence to id of its tokens
    """
    return [vocab.get(token, vocab['<unk>']) for token in tokenize(line)]

def convertDatasetFilesToTokenIds(data, mode):
    """ 
    Convert all the tokens in the data into their corresponding
    index in the vocabulary. 
    A file with same name _.ids will be created

    <s> will mark beginning of utterance and </s> marks end of utterance
    """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'rb')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'wb')
    
    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'dec': # we only care about '<s>' and </s> in encoder
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(stringToTokenIds(vocab, line))
        # ids.extend([vocab.get(token, vocab['<unk>']) for token in tokenize(line)])
        if mode == 'dec':
            ids.append(vocab['<\s>'])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

def load_data(enc_filename, dec_filename, max_training_size=None):
    """
    - Load questions and answers files. 
    - Each config.BUCKETS would collect QA pairs that conform to (question_max_size, answer_max_size) 
    -  For each question/answer pair, find the bucket tuple that they both belong to,
           break the for loop once found and go for next line

    returns data_buckets
    """
    encode_file = open(os.path.join(config.PROCESSED_PATH, enc_filename), 'rb')
    decode_file = open(os.path.join(config.PROCESSED_PATH, dec_filename), 'rb')
    encode, decode = encode_file.readline(), decode_file.readline()

    # each bucket is a tuple of (encode_max_size, decode_max_size) that should belong to same bucket
    # used for mini-batching
    #
    # For each bucket, create an empty array
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0

    # For each question/answer pair, find the bucket tuple that they both belong to,
    # break the for loop once found
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        
        # Get array of ids for question / answer pair
        encode_ids = [int(id_) for id_ in encode.split()] # Get array of ids that are in the question
        decode_ids = [int(id_) for id_ in decode.split()] # Get array of ids that are in the answer
        
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            # find question / answer pairs that comply to bucket string length limits
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets

def _pad_input(input_, size):
    """
    Pad a string up to maximum |size|
    """
    return input_ + [config.PAD_ID] * (size - len(input_))

def _reshape_batch(inputs, size, batch_size):
    """ 
    Create batch-major inputs. Batch inputs are just re-indexed inputs
    TODO
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks

#############
#############
#############
#############

print('Preparing raw data into train set and test set ...')
lineId2LineTextDictionary = getLineId2LineTextDictionary()
conversationsList = getConversationsList()
questions, answers = conversationToQuestionAnswerPairs(lineId2LineTextDictionary, conversationsList)
createTrainTestEncoderDecoderDataSets(questions, answers)

print('Preparing data to be model-ready ...')
build_vocab('train.enc')
build_vocab('train.dec')
convertDatasetFilesToTokenIds('train', 'enc')
convertDatasetFilesToTokenIds('train', 'dec')
convertDatasetFilesToTokenIds('test', 'enc')
convertDatasetFilesToTokenIds('test', 'dec')

for i in range(1,100):
    print (questions[i] + "    __________   " + answers[i])