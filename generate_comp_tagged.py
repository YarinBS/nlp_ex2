from create_embedding import generate_ds_comp, load_model
from data_parser import comp_parse_file, parse_file, parse_file_untagged
import pickle


def main():
    test_file_path = r'./data/test.untagged'
    test_file_tagged_path = r'./data/test.tagged.txt'
    model_path = r'./models/comp_svm_model.sav'
    windows_size = 3

    loaded_model = pickle.load(open(model_path, 'rb'))
    data = parse_file_untagged(test_file_path, windows_size)

    # tagged_data = parse_file(file_path=test_file_tagged_path,
    #                         windows_size=windows_size,
    #                         comp=False)

    x = generate_ds_comp(load_model(), data)

    predictions = loaded_model.predict(x)
    idx = 0
    with open(test_file_path, encoding='utf-8') as f:
        with open('data/test.tagged', 'a', encoding='utf-8') as f2:
            for row in f:
                if row == '\n':
                    f2.write('\n')
                else:
                    pred = predictions[idx]
                    idx += 1
                    if pred == 0:
                        f2.write(row.replace("\n", "") + "\t" + 'O' + "\n")
                    elif pred == 1:
                        f2.write(row.replace("\n", "") + "\t" + 'other_class' + "\n")
                    else:
                        print('Problem')


if __name__ == '__main__':
    main()
