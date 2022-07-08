import numpy as np
from numpy import savetxt
from numpy import loadtxt
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import pickle
from scipy.sparse.linalg import svds
from numpy.linalg import norm
from collections import Counter
import wordninja
import argparse


# Cosine distance
def cos_dis(u, v):
    dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
    return dist


def split_row_into_tokens(row):
    first_split = str(row[0]).split()
    result = []
    for t in first_split:
        result.extend( wordninja.split(t) )
    return result


def extract_and_save_tokens(df):
    print('Extracting vocubulary from the database of tokens...')
    vocabulary = Counter()
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        curr_tokens = str(row[0]).split()
        for token in curr_tokens:
            for subtoken in wordninja.split(token):
                vocabulary[subtoken] += 1

    voc = [x[0] for x in vocabulary.most_common()]
    voc_df = pd.DataFrame(voc)
    voc_df.to_csv('vocabulary.csv', index=False)  # , encoding='utf-8')
    with open('vocabulary.pickle', 'wb') as f:
        pickle.dump(voc, f)
    return voc


# Reading pickle back
def read_tokens():
    with open('vocabulary.pickle', 'rb') as f:
        voc = pickle.load(f)
    return voc


# Fill in cooccurence matrix
def fill_cooccur_mtx(voc, df):
    print('Filling in the cooccurrence matrix...')
    n = len(voc)
    cooccur_mtx = np.zeros((n, n))
    hash_map = {voc[i]: i for i in range(n)}
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        curr_tokens = split_row_into_tokens(row)
        curr_tokens_idxs = [hash_map[token] for token in curr_tokens]
        curr_tokens_num = len(curr_tokens_idxs)
        for i in range(curr_tokens_num):
            for j in range(i, curr_tokens_num):
                cooccur_mtx[curr_tokens_idxs[i], curr_tokens_idxs[j]] += 1
                cooccur_mtx[curr_tokens_idxs[j], curr_tokens_idxs[i]] += 1
    return cooccur_mtx


def save_cooccur_mtx(cooccur_mtx):
    savetxt('cooccur_mtx.csv', cooccur_mtx, delimiter=',')


def load_cooccur_mtx():
    return loadtxt('cooccur_mtx.csv', delimiter=',')


def save_embedding_mtx(w):
    savetxt('embedding_mtx.csv', w, delimiter=',')


def load_embedding_mtx():
    return loadtxt('embedding_mtx.csv', delimiter=',')


def find_closest_words(word, voc, w, closest_num=10):
    word_id = voc.index(word)
    distances = [ (j, cos_dis(w[word_id], w[j])) for j in range(len(voc)) ]
    min_dist_tokens = [voc[j] for i, (j, d) in enumerate(sorted(distances, key = lambda item: item[1])) if i < closest_num]
    result = f"List of closest words to {word}: " + ', '.join(min_dist_tokens)
    return word_id, result


def main(args):
    if args.build:
        df = pd.read_csv(args.df_path)
        voc = extract_and_save_tokens(df)  # extract tokens and build a vocabulary
        cooccur_mtx = fill_cooccur_mtx(voc, df)  # fill in the cooccurrence matrix
        # save_cooccur_mtx(cooccur_mtx)  # save the resulting cooccurrence matrix
        u, s, v = svds(cooccur_mtx, k=args.embed_dim)
        w = np.dot(u, np.diag(0.5 * s))  # extract the embedding matrix
        save_embedding_mtx(w)  # save the embedding matrix
    else:
        # cooccur_mtx = load_cooccur_mtx()  # load the cooccurrence matrix
        voc = read_tokens()
        w = load_embedding_mtx()  # load the embedding matrix
        _, result = find_closest_words(args.input_token, voc, w, args.closest_num)
        print(result)


parser = argparse.ArgumentParser(description='In the preprocessing regime (build is True): extract tokens from DF_PATH, \
                                build vocabulary, build token cooccurrence matrix, take its SVD to obtain token embeddings \
                                of dimension EMBED_DIM, save the results. Provide -p and -d arguments for that case. \
                                In the inference regime (build is False): find CLOSEST_NUM (default = 10) closest tokens to the given \
                                token INPUT_TOKEN. Provide -t and -l arguments.')
parser.add_argument('-b', '--build', action='store_true', default=False,
                    help='build and save vocabulary along with token embeddings (default: infer from the given token INPUT_TOKEN)')
parser.add_argument('-p', '--path', dest='df_path',
                    help='path to the csv file containing method names line by line')
parser.add_argument('-d', '--dim', dest='embed_dim', type=int,
                    help='embedding dimension')
parser.add_argument('-t', '--token', dest='input_token',
                    help='input token to make inference on')
parser.add_argument('-l', '--length', dest='closest_num', type=int, default=10,
                    help='length of the sequence of closest tokens')
args = parser.parse_args()

if __name__ == '__main__':
    main(args)