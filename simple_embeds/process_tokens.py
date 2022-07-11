import numpy as np
from numpy import savetxt
from numpy import loadtxt
import pandas as pd
from tqdm import tqdm
import pickle
from scipy.sparse.linalg import svds
from numpy.linalg import norm
from collections import Counter
import wordninja
import argparse
from word_mover_distance import model


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


def load_training_corpus():
    with open('labels.pickle', 'rb') as f:
        labels = pickle.load(f)
    return labels


def build_tag_vocabulary(labels):
    print('Building a vocabulary of tags...')
    tags = set()
    for label in tqdm(labels, total=len(labels)):
        tags.add(' '.join(label))
    result = list(tags)
    with open('tags.pickle', 'wb') as f:
        pickle.dump(result, f)
    print(f'A tag vocabulary of length {len(result)} has been built and saved.')
    print('')
    return result


def load_tag_vocabulary():
    with open('tags.pickle', 'rb') as f:
        tags = pickle.load(f)
    return tags


def initialize_wmd_model(voc, w):
    word_embeds = {voc[i]: w[i] for i in range(len(voc))}
    wmd_model = model.WordEmbedding(model=word_embeds)
    return wmd_model


def find_wmd_distance(wmd_model, label1, label2):
    return wmd_model.wmdistance(label1.split(), label2.split())


def find_closest_labels(label, tags, wmd_model, closest_num=10):
    distances = [ (j, find_wmd_distance(wmd_model, label, tags[j])) for j in range(len(tags)) ]
    min_dist_names = [ f"'{tags[j]}'" for i, (j, d) in enumerate(sorted(distances, key = lambda item: item[1])) if i < closest_num ]
    print(f"List of closest labels to '{label}': " + ', '.join(min_dist_names))


def main(args):
    if args.build:
        df = pd.read_csv(args.df_path)
        voc = extract_and_save_tokens(df)  # extract tokens and build a vocabulary
        cooccur_mtx = fill_cooccur_mtx(voc, df)  # fill in the cooccurrence matrix
        # save_cooccur_mtx(cooccur_mtx)  # save the resulting cooccurrence matrix
        u, s, v = svds(cooccur_mtx, k=args.embed_dim)
        w = np.dot(u, np.diag(0.5 * s))  # extract the embedding matrix
        save_embedding_mtx(w)  # save the embedding matrix
        labels = load_training_corpus()  # load the database of labels
        tags = build_tag_vocabulary(labels)  # build a vocabulary of tags, i.e. unique labels
    elif args.label:
        tags = load_tag_vocabulary()
        voc = read_tokens()  # load the vocabulary
        w = load_embedding_mtx()  # load the embedding matrix
        wmd_model = initialize_wmd_model(voc, w)
        find_closest_labels(args.input, tags, wmd_model, args.closest_num)
    elif args.labeld:
        voc = read_tokens()  # load the vocabulary
        w = load_embedding_mtx()  # load the embedding matrix
        wmd_model = initialize_wmd_model(voc, w)
        print(find_wmd_distance(wmd_model, args.input1, args.input2))
    else:
        # cooccur_mtx = load_cooccur_mtx()  # load the cooccurrence matrix
        voc = read_tokens()  # load the vocabulary
        w = load_embedding_mtx()  # load the embedding matrix
        _, result = find_closest_words(args.input, voc, w, args.closest_num)
        print(result)


parser = argparse.ArgumentParser(description='In the preprocessing regime (BUILD is True): extract tokens from DF_PATH, \
                                build vocabulary, build token cooccurrence matrix, take its SVD to obtain token embeddings \
                                of dimension EMBED_DIM, build tag vocabulary, save the results. Provide -p and -d parameters for that case. \
                                In the inference regime (BUILD is False): find CLOSEST_NUM (default = 10) closest tokens to the given \
                                token INPUT (provide -i and -n parameters); if LABEL is true, then find CLOSEST_NUM (default = 10) \
                                labels to the given label INPUT (provide -i and -n parameters); if LABELD is true, then find the WMD \
                                distance between the labels INPUT1 and INPUT2 (provide -i1 and -i2 parameters).')
parser.add_argument('-b', '--build', action='store_true', default=False,
                    help='build and save vocabulary along with token embeddings and tags (default: find closest tokens to the given INPUT_TOKEN)')
parser.add_argument('-l', '--label', action='store_true', default=False,
                    help="find closest labels to the given one using the Word Mover's Distance (default: find closest tokens to the given INPUT_TOKEN)")
parser.add_argument('-ld', '--labeld', action='store_true', default=False,
                    help="find the distance between two labels INPUT1 and INPUT2 using the Word Mover's Distance (default: find closest tokens to the given INPUT_TOKEN)")
parser.add_argument('-p', '--path', dest='df_path',
                    help='path to the csv file containing method names line by line')
parser.add_argument('-d', '--dim', dest='embed_dim', type=int,
                    help='embedding dimension')
parser.add_argument('-i', '--input', dest='input',
                    help='input token or label')
parser.add_argument('-i1', '--input1', dest='input1',
                    help='first input label')
parser.add_argument('-i2', '--input2', dest='input2',
                    help='second input label')
parser.add_argument('-n', '--clnum', dest='closest_num', type=int, default=10,
                    help='length of the returned sequence of closest tokens / labels')
args = parser.parse_args()

if __name__ == '__main__':
    main(args)

#exec(open("process_tokens.py").read())

# Some results:
# List of closest labels to 'get token position': 'get token position', 'get label position', 'get token offset', 'get location token', 'get token location', 'get row position', 'get position parent', 'get parent position', 'get icon position', 'get token icon'
# List of closest labels to 'build word tree': 'build symbol tree', 'build execution tree', 'build final tree', 'build binary tree', 'build debug tree', 'build leaf tree', 'build dependency tree', 'build interval tree', 'build hierarchy tree', 'build composite tree'