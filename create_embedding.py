from gensim import downloader, models
import numpy as np
import os
from gensim.models import KeyedVectors
from gensim.downloader import base_dir
import pickle
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from parser import parse_file


def load_model():
    # WORD_2_VEC_PATH = 'word2vec-google-news-300'
    GLOVE_PATH = 'glove-twitter-200'
    glove = downloader.load(GLOVE_PATH)
    return glove


def embed(model, sen):
    representation = []
    for word in sen:
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


def train(train_set, validation_set):
    glove_model = load_model()
    x, y = generate_ds(glove_model, train_set)
    clf = svm.SVC()
    clf.fit(x, y)

    x_validation, y_validation = generate_ds(glove_model, validation_set)
    y_validation_pred = clf.predict(x_validation)

    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = f1_score(y_validation, y_validation_pred, average='macro')
    print(f'f1 score is {f1}')

    filename = 'svm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # some time later...

    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)


def main():
    train_file_path = r'./data/train.tagged'
    validation_file_path = r'./data/dev.tagged'
    windows_size = 2
    train_set = parse_file(file_path=train_file_path,
                           windows_size=windows_size)

    validation_set = parse_file(file_path=validation_file_path,
                                windows_size=windows_size)

    # train_set = train_set[:1000]
    # validation_set = validation_set[:1000]
    train(train_set, validation_set)


if __name__ == '__main__':
    main()
