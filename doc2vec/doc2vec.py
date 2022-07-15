from this import d
from attr import asdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import wordninja
from numpy.linalg import norm
from numpy import savetxt
from numpy import loadtxt
from collections import Counter


VECTOR_SIZE = 100


# Cosine distance
def cos_dis(u, v):
    dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
    return dist


def read_and_split_labels():
    print('Reading and splitting the method labels...')
    df = pd.read_csv('/hdd/rmusab/names_training.csv')
    labels = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        label = []
        for token in str(row[0]).split():
            label.extend(wordninja.split(token))
        #labels.append(str(row[0]).split())
        labels.append(label)
    with open('labels.pickle', 'wb') as f:
        pickle.dump(labels, f)
    print('')
    return labels


def load_training_corpus():
    with open('labels.pickle', 'rb') as f:
        labels = pickle.load(f)
    return labels


def train_and_save_doc2vec():
    # Set up logging to track the training progress
    logging.basicConfig(filename='gensim.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.INFO)
    # Train doc2vec model
    print('Training the doc2vec model...')
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(labels)]
    model = Doc2Vec(documents, vector_size=VECTOR_SIZE, window=6, min_count=1, workers=8, epochs=40)
    # Save model
    model.save('my_doc2vec_model3')
    print('The model has been saved')
    return model


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


def convert_tags_to_vectors(labels, model):
    print('Converting the set of tags to embedded vectors...')
    n = len(labels)
    tags_vec = np.zeros((n, VECTOR_SIZE))
    for i, label in tqdm(enumerate(labels), total=n):
        tags_vec[i] = model.infer_vector(label.split())
    savetxt('tag_vectors.csv', tags_vec, delimiter=',')  # save tag vectors
    print('')
    return tags_vec


def load_tags_vectors():
    return loadtxt('tag_vectors.csv', delimiter=',')


def get_closest_names(name, tags, model, tags_vec, closest_num=10):
    vector = model.infer_vector(name)
    distances = [ (j, cos_dis(vector, tags_vec[j])) for j in range(len(tags_vec)) ]
    min_dist_names = [ f"'{tags[j]}'" for i, (j, d) in enumerate(sorted(distances, key = lambda item: item[1])) if i < closest_num ]
    print(f"List of closest words to '{' '.join(name)}': " + ', '.join(min_dist_names))


def extract_and_save_tokens(labels, model):
    print('Extracting vocabulary from the database of labels...')
    vocabulary = Counter()
    for i, label in tqdm(enumerate(labels), total=len(labels)):
        for token in label:
            vocabulary[token] += 1

    voc = [x[0] for x in vocabulary.most_common()]
    with open('vocabulary.pickle', 'wb') as f:
        pickle.dump(voc, f)
    n = len(voc)
    tokens_vec = np.zeros((n, VECTOR_SIZE))
    for i, token in enumerate(voc):
        token_vec = model.infer_vector([token])
        tokens_vec[i] = token_vec
    savetxt('token_vectors.csv', tokens_vec, delimiter=',')  # save token vectors
    return voc, tokens_vec


def load_tokens_and_vec():
    with open('vocabulary.pickle', 'rb') as f:
        voc = pickle.load(f)
    tokens_vec = loadtxt('token_vectors.csv', delimiter=',')
    return voc, tokens_vec


def get_new_closest_tokens(model, target_vec, curr_label, token_voc, length=3):
    n = len(token_voc)
    distances = [ (j, cos_dis(target_vec, model.infer_vector(curr_label + [token_voc[j]]))) for j in range(n) ]
    min_dist_tokens = [ token_voc[j] for i, (j, d) in enumerate(sorted(distances, key = lambda item: item[1])) if i < length ]
    return min_dist_tokens


def generate_close_labels(target_label, model, token_voc, length=3):
    curr_labels = [[]]
    target_vector = model.infer_vector(target_label)
    for h in range(length):
        print(curr_labels)
        new_labels = []
        for curr_label in curr_labels:
            min_dist_tokens = get_new_closest_tokens(model, target_vector, curr_label, token_voc, length)
            for new_token in min_dist_tokens:
                new_labels.append(curr_label + [new_token])
        curr_labels = new_labels[:]
    return curr_labels


#labels = read_and_split_labels()
labels = load_training_corpus()

#model = train_and_save_doc2vec()
model = Doc2Vec.load("my_doc2vec_model3")  # load saved model

#tags = build_tag_vocabulary(labels)
#tags = load_tag_vocabulary()

#tags_vec = convert_tags_to_vectors(tags, model)
#tags_vec = load_tags_vectors()

#token_voc, tokens_vec = extract_and_save_tokens(labels, model)
token_voc, tokens_vec = load_tokens_and_vec()

#name1 = ["set", "property", "name"]
#name2 = ["get", "var", "cnt"]
#name2 = ["get", "dimension", "count"]
#vector1 = model.infer_vector(name1)
#vector2 = model.infer_vector(name2)
#print(f"Cosine distance between the names '{' '.join(name1)}' and '{' '.join(name2)}': {cos_dis(vector1, vector2)}")

# Get closest_num closest labels to name
#name = ["build", "index"]
#get_closest_names(name, tags, model, tags_vec)

close_labels = generate_close_labels(['get', 'token', 'value'], model, token_voc)
for label in close_labels:
    print(' '.join(label))

#List of closest words to 'get var count': 'is empty', 'swipe left', 'add var bind', 'update snitch', 'compress image', 'get world center', 'at for s tmnt', 'child count', 'get var count', 'show loading'
#exec(open("doc2vec.py").read())

# Some ways to improve:
# 1. Learn a language model and sample from it while taking into account the distance to the target label in our doc2vec embedding space
# 2. Extract learned word embeddings from doc2vec and compare them to those generated by the SVD model
# 3. Include the source code into context window for embeddings
# 4. See how the doc2vec metric compares to, say, our SVD embeddings equipped with the Word Mover's Distance
# 5. Investigate why generate_close_labels method doesn't work well
# 6. Maybe just use pretrained good databases of synonyms, like WordNet, for equivalent method name generation?
# 7. Train Doc2Vec using a DBOW model (dm=0)