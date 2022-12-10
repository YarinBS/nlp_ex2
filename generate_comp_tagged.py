from create_embedding import generate_ds_comp, load_model
from data_parser import comp_parse_file
import pickle


def main():
    test_file_path = r'./data/test.untagged'
    model_path = r'./models/comp_svm_model.sav'
    windows_size = 2

    loaded_model = pickle.load(open(model_path, 'rb'))
    data = comp_parse_file(test_file_path, windows_size)

    x = generate_ds_comp(load_model(), data)

    predictions = loaded_model.predict(x)

    with open(test_file_path, encoding='utf-8') as f:
        with open('test.tagged', 'a', encoding='utf-8') as f2:
            for word, pred in zip(f, predictions):
                f2.write(word.replace("\n", "") + "\t" + str(pred) + "\n")


if __name__ == '__main__':
    main()
