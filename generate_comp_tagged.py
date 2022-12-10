from ff_model import Net
from torch.utils.data import DataLoader
from create_embedding import generate_ds, generate_ds_comp, load_model
from data_parser import comp_parse_file
import torch
from sklearn.metrics import f1_score
import pickle


def main():
    test_file_path = r'./data/test.untagged'
    model_path = r'./models/svm_model.sav'
    windows_size = 2

    loaded_model = pickle.load(open(model_path, 'rb'))
    data = comp_parse_file(test_file_path, windows_size)

    x = generate_ds_comp(load_model(), data)

    predictions = loaded_model.predict(x)

    with open(test_file_path) as f:
        with open('test.tagged', 'a') as f2:
            for word, pred in zip(f, predictions):
                f2.write(f'{word}\t{pred}\n')


if __name__ == '__main__':
    main()
