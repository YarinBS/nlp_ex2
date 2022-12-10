from data_parser import parse_file
from create_embedding import load_model, generate_ds
from sklearn import svm
from sklearn.metrics import f1_score
import pickle


def train(train_set, validation_set):
    glove_model = load_model()
    x, y = generate_ds(glove_model, train_set)
    clf = svm.SVC()
    clf.fit(x, y)

    x_validation, y_validation = generate_ds(glove_model, validation_set)
    y_validation_pred = clf.predict(x_validation)

    f1 = f1_score(y_validation, y_validation_pred, average='macro')
    print(f'f1 score is {f1}')

    filename = 'comp_svm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)


def main():
    train_file_path = r'./data/train.tagged'
    validation_file_path = r'./data/dev.tagged'
    windows_size = 2
    train_set = parse_file(file_path=train_file_path,
                           windows_size=windows_size,
                           train=True)

    validation_set = parse_file(file_path=validation_file_path,
                                windows_size=windows_size,
                                train=False)

    # train_set = train_set[:1000]
    # validation_set = validation_set[:1000]
    train(train_set, validation_set)


if __name__ == '__main__':
    main()
