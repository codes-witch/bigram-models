import argparse
import random
import math
import pandas as pd
import numpy as np

BOS_MARKER = "<s>"
EOS_MARKER = "</s>"
UNKNOWN = "<UNK>"


# Generate a sentence with probabilities according
# to the given bigram model
# The returned sentence is a list of words ['<s>', 'w1', ..., 'wn', '</s>']
# Replace UNKNOWN with a randomly selected word from unknown_words.
def generate_sentence(prob_df, unknown_words):
    sentence = ['<s>']

    # Init basic sentence.
    last_word = sentence[-1]
    # Add a random word based on bigram weights until end token is generated
    while 1:
        sentence += random.choices(prob_df.columns, prob_df.loc[last_word])
        last_word = sentence[-1]

        if last_word == '</s>': break
        # Replace UNK token with randomly selected word
        if last_word == "<UNK>": sentence[-1] = random.choice(unknown_words)

    return sentence


# Return the log probability of sent,
# where sent is a list of words [<s>, 'w1', ..., 'wn', </s>].
def get_sent_logprob(log_df, sent):
    log_prob = 0
    for index in range(1, len(sent)): log_prob += log_df.loc[sent[index - 1], sent[index]]

    return log_prob


# Calculate the perplexity of the model with the given test sentences
def get_perplexity(log_df, test_sentences):
    # Extract log probabilities, Calculate Perplexity of all sentences appearing in concession
    corpus_logprob = sum([get_sent_logprob(log_df, sent) for sent in test_sentences])

    # Get number of words in corpues, counting EOS tokens, omitting BOS tokens
    n_words = sum([len(sent) for sent in test_sentences]) - len(test_sentences)

    # Assigning variables to perplexity formula
    return math.pow(10, (-1 / n_words) * corpus_logprob)


# Return a DataFrame representing the unigram counts for the
# tokenized training data.
def get_unigram_counts(training_data):
    # counts is a dict: {key=word, value=count}
    counts = {}

    for sent in training_data:
        for tok in sent:
            counts[tok] = counts.get(tok, 0) + 1

    # Convert dict to dataframe and return the dataframe
    count_df = pd.DataFrame(counts, index=[0])
    return count_df


# Replace words that only appear once in the training data with UNKNOWN.
# This is done by by generating unigram counts, then combining into one
# column (with the special name UNKNOWN), all columns with count==1.
# Return the vocabulary of the training data and a list of unknown words.
def replace_oov(training_data):

    count_df = get_unigram_counts(training_data)

    # Words that appear only once are considered UNKNOWN
    # Get the column headers (words) with count == 1
    row0 = count_df.iloc[0]
    unknown_words = row0.index.values[row0 == 1]
    num_unknown_words = len(unknown_words)

    # drop unknown words columns
    count_df.drop(unknown_words, axis=1, inplace=True)

    # Create an UNKNOWN column (with sum of unknown words)
    count_df[UNKNOWN] = num_unknown_words

    # Vocabulary is a list of the column names
    vocab = list(count_df.columns)

    return vocab, unknown_words


# Return a dataframe of bigram counts, where
# rows represent w1 and columns represent w2.
# If a vocabulary is given, replace words that are not in the vocabulary with UNKNOWN.
# If padding=True, insert BOS and EOS sentence markers.
def get_bigram_counts(training_data, vocab=None, padding=True):

    # counts is a nested dict: {key=w_i, value = {key=w_j, value=count}}
    counts = {}

    for sentence in training_data:
        if vocab:
            sent = [w if w in vocab else UNKNOWN for w in sentence]
        else:
            sent = sentence

        if padding:
            sent.insert(0, BOS_MARKER)
            sent.append(EOS_MARKER)

        # iterate sent bigrams: (w1,w2), (w2,w3), (w3,w4)...
        for (w_i, w_j) in zip(sent, sent[1:]):
            counts[w_i] = counts.setdefault(w_i, {w_j: 0})
            counts[w_i][w_j] = counts[w_i].setdefault(w_j, 0) + 1

    count_df = pd.DataFrame(counts).T
    count_df.replace(np.nan, 0, inplace=True)
    return count_df


# Convert all dataframe cells to log probabilities using base 10
def df_to_log(prob_df):
    return np.log10(prob_df)


# Read corpus file line-by-line and perform preprocessing
# Returns a list of tokenized sentences
# If vocab is provided, oov words are replaced with UNKNOWN
# If padding=True, add BOS and EOS sentence markers
def read_and_preprocess_corpus(corpus_file, vocab=None, padding=False):
    with open(corpus_file, "r", encoding="utf-8") as corpus:
        sentences = corpus.readlines()

    tok_sentences = []
    for sent in sentences:
        tok_sent = sent.lower()
        tok_sent = tok_sent.split()
        if vocab:
            tok_sent = [tok if tok in vocab else UNKNOWN for tok in tok_sent]
        if padding:
            tok_sent.insert(0, BOS_MARKER)
            tok_sent.append(EOS_MARKER)
        tok_sentences.append(tok_sent)

    return tok_sentences


#  Build Kneiser ney model using the given bigram counts dataframe.
def build_model(bigram_counts, d=.75):
    # Get total bigram counts.
    non_zeroes = 0
    for column in bigram_counts.columns:
        non_zeroes += sum(bigram_counts[column] != 0)

    # Extract Bigram count to dict for easy access.
    w1_to_w2_bigram_counts = bigram_counts.transpose().to_dict()

    # Get word and bigram counts.
    total_start_end_counts = get_total_start_end_pcont(bigram_counts, w1_to_w2_bigram_counts, non_zeroes)

    # Perform Kneser Ney smoothing for each cell.
    kn_dict = dict()
    # wi is word1, wj is word2
    for w_i in bigram_counts.index:
        for w_j in bigram_counts.columns:
            kn_dict[w_i] = kn_dict.setdefault(w_i, {w_j: 0})
            kn_dict[w_i][w_j] = kn_dict[w_i].setdefault(w_j, 0)
            kn_dict[w_i][w_j] = kneser_ney(d, w1_to_w2_bigram_counts[w_i][w_j], total_start_end_counts[w_i][0],
                                            total_start_end_counts[w_i][1], total_start_end_counts[w_j][2])

    kn_df = pd.DataFrame(kn_dict)
    kn_df.replace(np.nan, 0, inplace=True)

    return kn_df.transpose()


def get_total_start_end_pcont(bigram_counts, w1_to_w2_bigram_counts, total_bigram_counts):
    """
     Create a dictionary for every word
     with its counts, number of bigram types it starts,
     number of bigram types it ends
    :returns: a dictionary, Key - word. Value- [<counts>, <bigram types it starts>, <pcont_w>]
    """
    total_start_pconts = dict().fromkeys(list(bigram_counts.columns) + ["<s>"])

    # Init value array as placeholder for Total, bigram start and bigram end counts.
    for w1 in total_start_pconts.keys(): total_start_pconts[w1] = [0.0, 0.0, 0.0]

    for w1 in w1_to_w2_bigram_counts.keys():
        for w2 in w1_to_w2_bigram_counts[w1].keys():
            if w1_to_w2_bigram_counts[w1][w2] != 0.0:
                # word counts (index 0)
                total_start_pconts[w1][0] += w1_to_w2_bigram_counts[w1][w2]
                total_start_pconts[w1][1] += 1.0  # bigram types it starts (index 1)
                total_start_pconts[w2][2] += 1.0  # bigram types it ends (index 2)
    total_start_pconts["</s>"][0] = total_start_pconts["<s>"][0]

    # Obtain pcont from Bigram end counts.
    for w1 in total_start_pconts.keys(): total_start_pconts[w1][2] /= total_bigram_counts

    return total_start_pconts


# Perform kneser ney smoothing on given cell according to input.
def kneser_ney(d, bigram_count, w1_count, wtypes_start_w1, pcont_w2):
    lambda_w1 = (d * wtypes_start_w1) / w1_count

    return (max(bigram_count - d, 0) / w1_count) + (lambda_w1 * pcont_w2)


# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_file", help="corpus file")
    parser.add_argument("test_file", help="test file")
    return parser.parse_args()


def main(args):
    # Read the training data
    train = read_and_preprocess_corpus(args.corpus_file)

    # Replace OOV words, and get the training vocab and unknown words
    vocab, unknown_words = replace_oov(train)

    # Read the test corpus, using the fixed vocabulary, and adding sentence markers
    test_sentences = read_and_preprocess_corpus(args.test_file, vocab=vocab, padding=True)

    # Calculate and print the OOV rate of the test corpus
    n_words = 0
    unks = 0

    for sent in test_sentences:
        n_words += len(sent) - 2  # remove BOS and EOS
        unks += sent.count(UNKNOWN)

    oov_rate = unks / n_words * 100
    print(f'OOV rate test: {oov_rate}')

    # Get the bigram counts, using the fixed vocabulary
    bigram_counts = get_bigram_counts(train, vocab=vocab)

    # Build Kneiser Ney model
    bigram_model = build_model(bigram_counts)

    # convert probabilities to log probabilities
    log_df = df_to_log(bigram_model)

    # Generate and print 10 sentences using the model
    for i in range(0, 10):
        # use probabilities to generate sentences
        sent = generate_sentence(bigram_model, unknown_words)
        sent = ' '.join(sent)
        print(f'{sent}')

    # Print the sentence probabilities for the first 5 test sentences
    for test_sent in test_sentences[:5]:
        # use log probabilities to get prob of a sentence
        logprob = get_sent_logprob(log_df, test_sent)
        print(f'P({test_sent}): {math.pow(10, logprob)}')

    # Calculate the perplexity of the test sentences' corpus
    # Use log probabilities to calculate perplexity
    perplexity = get_perplexity(log_df, test_sentences)
    print(f'\nperplexity of test_sentences: {perplexity}')


if __name__ == '__main__':
    main(parse_args())