from nltk.tag import hmm
from nltk.probability import FreqDist
from nltk.probability import LaplaceProbDist
from nltk.probability import ConditionalProbDist
from nltk.probability import ConditionalFreqDist
from nltk.probability import MLEProbDist
from nltk.tag import HiddenMarkovModelTagger
import string
import re
import argparse

def extra_text():
    lower_list = list(string.ascii_lowercase)
    symbols = [' ', ',', '.']
    with open('rt-polarity.pos', encoding="ISO-8859-1") as f:
        raw = f.read()
        f.close()
    sequences = raw.split('\n')
    with open('rt-polarity.neg', encoding="ISO-8859-1") as f:
        raw1 = f.read()
        f.close()
    sequences1 = raw1.split('\n')

    sequences1 = sequences+sequences1

    # lower case
    sequences2 = []
    for sequence in sequences1:
        sequences2.append(sequence.lower())
    # remove chars
    sequences3 = []
    for sequence in sequences2:
        new_seq = ''
        for char in sequence:
            if (char in lower_list) or (char in symbols):
                new_seq += char
        sequences3.append(new_seq)
    # print(sequences3)

    # remove extra spaces
    sequences4 = []
    for sequence in sequences3:
        seq = re.sub(r'\s([?.,!"](?:\s|$))', r'\1', sequence)
        seq = seq.strip()
        sequences4.append(seq)

    return [[char for char in seq] for seq in sequences4]


def train_supervised2(trainer, labelled_sequences, plain_sequences, estimator=None):
    _TAG = 1
    _TEXT = 0
    if estimator is None:
        estimator = lambda fdist, bins: MLEProbDist(fdist)

        # count occurrences of starting states, transitions out of each state
        # and output symbols observed in each state
    known_symbols = set(trainer._symbols)
    known_states = set(trainer._states)

    starting = FreqDist()
    transitions = ConditionalFreqDist()
    outputs = ConditionalFreqDist()
    # =================code added to supplement transition matrix====================
    for sequence in plain_sequences:
        lasts = None
        for token in sequence:
            if lasts is None:
                pass
            else:
                transitions[lasts][token] += 1
            lasts = token

            if token not in known_states:
                trainer._states.append(token)
                known_states.add(token)
    # ================================end============================================
    for sequence in labelled_sequences:
        lasts = None
        for token in sequence:
            state = token[_TAG]
            symbol = token[_TEXT]
            if lasts is None:
                starting[state] += 1
            else:
                transitions[lasts][state] += 1
            outputs[state][symbol] += 1
            lasts = state

            # update the state and symbol lists
            if state not in known_states:
                trainer._states.append(state)
                known_states.add(state)

            if symbol not in known_symbols:
                trainer._symbols.append(symbol)
                known_symbols.add(symbol)

    # create probability distributions (with smoothing)
    N = len(trainer._states)
    pi = estimator(starting, N)
    A = ConditionalProbDist(transitions, estimator, N)
    B = ConditionalProbDist(outputs, estimator, len(trainer._symbols))

    return HiddenMarkovModelTagger(trainer._symbols, trainer._states, A, B, pi)



# return a list of char in a string
def split(string):
    return [char for char in string]


# Transform a tuple list to a string
def tuple_list2deciphered_string(tuple_list, index):
    return ''.join([tpl[index] for tpl in tuple_list])


def get_text(path):
    with open(path) as f:
        # list of strings
        ret = f.read().split('\n')
        f.close()
    return ret

# return list(list(tuple(character, character)))
def format_data(cipher, plain):
    len_min = min(len(cipher), len(plain))
    ret = []
    for i in range(len_min):
        cipher_str = cipher[i]
        plain_str = plain[i]
        sub_ret = []
        for j in range(len(cipher_str)):
            sub_ret.append((cipher_str[j], plain_str[j]))
        ret.append(sub_ret)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='path in which the training data locate')

    parser.add_argument('-laplace', '--laplace_smoothing', help='Turn laplace smoothing on', action='store_true')
    parser.add_argument('-lm', '--supplement', help='Add supplement document to update transition matrix', action='store_true')
    args = parser.parse_args()
    cipher_folder = args.folder

    # training data
    plain_path = cipher_folder + '/train_plain.txt'
    cipher_path = cipher_folder + '/train_cipher.txt'
    cipher_train = get_text(cipher_path)
    plain_train = get_text(plain_path)
    # format the training data
    train_data = format_data(cipher_train, plain_train)


    # test data
    testc_path = cipher_folder + '/test_cipher.txt'
    testp_path = cipher_folder + '/test_plain.txt'
    testc = get_text(testc_path)
    testp = get_text(testp_path)
    # format the test data
    test_data = format_data(testc, testp)

    trainer = hmm.HiddenMarkovModelTrainer()

    #laplace estimator
    my_estimator = lambda fdist, bins: LaplaceProbDist(fdist, bins)

    if args.laplace_smoothing:
        if args.supplement:
            tagger = train_supervised2(trainer, train_data, extra_text(), estimator=my_estimator)
        else:
            tagger = trainer.train_supervised(train_data, estimator=my_estimator)
    else:
        if args.supplement:
            tagger = train_supervised2(trainer, train_data, extra_text())
        else:
            tagger = trainer.train_supervised(train_data)

    print(tagger.evaluate(test_data))
    for sentence in testc:
        print(tuple_list2deciphered_string(tagger.tag(split(sentence)), 1))





