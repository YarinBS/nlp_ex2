from gensim import downloader, models
import numpy as np
import os
from gensim.models import KeyedVectors
from gensim.downloader import base_dir
import pickle
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def load_model():
    # WORD_2_VEC_PATH = 'word2vec-google-news-300'
    GLOVE_PATH = 'glove-twitter-200'
    glove = downloader.load(GLOVE_PATH)
    return glove


def embed(model, sen):
    representation = []
    for word in sen.split():
        if word not in model.key_to_index:
            print(f"The word: [{word}] is not an existing word in the model")
            vec = np.zeros(200)
        else:
            vec = model[word]
        representation = representation + list(vec)
    representation = np.asarray(representation)
    return representation


def main(ds):
    glove_model = load_model()
    ds = [("am a student at the", 0), ("i ate sandwich yesterday and", 1)]
    x = []
    y = []
    for (sample, label) in ds:
        sentence_embeddings = embed(glove_model, sample)
        x.append(sentence_embeddings)
        y.append(label)

    clf = svm.SVC()
    clf.fit(x, y)
    sentence_embeddings = embed(glove_model, "I am hatoceleb here now")
    out = clf.predict([sentence_embeddings])
    print('Predictions:')
    print(out)

    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 1, 0]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'f1 score is {f1}')

    filename = 'svm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # some time later...

    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)




if __name__ == '__main__':
    main('')
