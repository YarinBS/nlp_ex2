from gensim import downloader
import numpy as np


def load_model():
    # WORD_2_VEC_PATH = 'word2vec-google-news-300'
    GLOVE_PATH = 'glove-twitter-200'
    glove = downloader.load(GLOVE_PATH)
    return glove


def embed(model, sen):
    representation = []
    for word in sen:
        word = word.lower()
        if word not in model.key_to_index:
            print(f"The word: [{word}] is not an existing word in the model")
            vec = np.zeros(200)
        else:
            vec = model[word]
        representation = representation + list(vec)
    representation = np.asarray(representation)
    return representation


def generate_ds(glove_model, string_data):
    x = []
    y = []
    for (sample, label) in string_data:
        sentence_embeddings = embed(glove_model, sample)
        x.append(sentence_embeddings)
        y.append(label)
    return x, y


def generate_ds_comp(glove_model, string_data):
    x = []
    for sample in string_data:
        sentence_embeddings = embed(glove_model, sample)
        x.append(sentence_embeddings)
    return x
