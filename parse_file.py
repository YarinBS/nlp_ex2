from gensim import downloader
import numpy as np


def load_model():
    # WORD_2_VEC_PATH = 'word2vec-google-news-300'
    GLOVE_PATH = 'glove-twitter-200'
    glove = downloader.load(GLOVE_PATH)
    glove.save("glove.model")

    return glove


def embed(model, sen):
    representation = []
    for word in sen.split():
        if word not in model.key_to_index:
            print(f"{word} not an existing word in the model")
            continue
        vec = model[word]
        representation.append(vec)
    representation = np.asarray(representation)
    return representation


def main():
    sen = "i am a student at the technion"
    glove_model = load_model()
    vec = embed(glove_model, sen)
    print(vec)

if __name__ == '__main__':
    main()
