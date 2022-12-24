from data_parser import parse_file
from create_embedding import load_model, generate_ds
from sklearn import svm
from sklearn.metrics import f1_score
import pickle


def train(train_set, validation_set, test_set):
    glove_model = load_model()
    x, y = generate_ds(glove_model, train_set)
    clf = svm.SVC()
    clf.fit(x, y)

    x_validation, y_validation = generate_ds(glove_model, validation_set)
    x_test, y_test = generate_ds(glove_model, test_set)
    y_validation_pred = clf.predict(x_validation)
    y_test_pred = clf.predict(x_test)

    validation_f1 = f1_score(y_validation, y_validation_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    print(f'validation set: f1 score is {validation_f1}')
    print(f'test set: f1 score is {test_f1}')

    filename = 'models/comp_svm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)


def main():
    real_test_set_path = r'./data/test.tagged.txt'
    generated_test_set_path = r'./data/test.tagged'
    windows_size = 3
    real_test_set = parse_file(file_path=real_test_set_path,
                               windows_size=windows_size,
                               comp=False)

    generated_test_set = parse_file(file_path=generated_test_set_path,
                                    windows_size=windows_size,
                                    comp=False)
    real_test_labels = [y for x,y in real_test_set]
    generated_test_labels = [y for x,y in generated_test_set]
    test_f1 = f1_score(real_test_labels, generated_test_labels)
    print(f'f1 of generated test set: {test_f1:.3f}')


if __name__ == '__main__':
    main()
