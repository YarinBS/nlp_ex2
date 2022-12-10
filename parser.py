"""
Return a list containing tuples,
each contains a target word and its context window
for a given sentence
(sentence, num)
num = 0 if the center word does not have a label
else num = 1
"""

from string import punctuation


def parse_file(file_path, windows_size,train):
    with open(file_path) as f:

        sentences = []  # Contains the final sentences without tags
        sentence_tags = []  # Contains the tags of each word in the sentences
        new_sentence = []
        new_sentence_tags = []
        for row in f:
            # todo: deal with end of file dev.tagged!!!!
            if row != '\t\n' and row != '\n' and row != '\ufeff':  # If still in the current sentence:
                word_to_add = row.split("\t")[0].lower()
                tag_to_add = row.split("\t")[1].replace('\n', '')
                new_sentence.append(word_to_add)
                new_sentence_tags.append(tag_to_add)
            else:
                sentences.append(new_sentence)
                sentence_tags.append(new_sentence_tags)
                new_sentence, new_sentence_tags = [], []

        for sentence, tags in zip(sentences, sentence_tags):
            # Adding astericks padding to the sentence
            # Beginning of sentence:
            for j in range(windows_size):
                sentence.insert(0, '*')
                tags.insert(0, '*')

            # Sentence ending:
            for j in range(windows_size):
                sentence.append('*')
                tags.append('*')

        dataset = []
        for sen, tags in zip(sentences, sentence_tags):
            for i in range(windows_size, len(sen) - windows_size):
                # Checking the tuple's tag
                if tags[i] == 'O':
                    tuple_tag = 0
                else:
                    tuple_tag = 1

                # Creating the context of the current word
                words_in_the_tuple = []
                for j in range(i - windows_size, i + windows_size + 1):
                    words_in_the_tuple.append(sen[j])

                dataset.append([words_in_the_tuple, tuple_tag])

    if train:
        extended_ds = []
        for sen, tag in dataset:
            if tag == 1:
                for i in range(15):
                    extended_ds.append((sen, tag))
            else:
                extended_ds.append((sen, tag))
        return extended_ds
    else:
        return dataset


def comp_parse_file(file_path):
    with open(file_path) as f:

        sentences = []  # Contains the final sentences without tags
        sentence_tags = []  # Contains the tags of each word in the sentences
        new_sentence = []
        new_sentence_tags = []
        for row in f:
            # todo: deal with end of file dev.tagged!!!!
            if row != '\t\n' and row != '\n' and row != '\ufeff':  # If still in the current sentence:
                word_to_add = row.split("\t")[0].lower()
                tag_to_add = row.split("\t")[1].replace('\n', '')
                new_sentence.append(word_to_add)
                new_sentence_tags.append(tag_to_add)
            else:
                sentences.append(new_sentence)
                sentence_tags.append(new_sentence_tags)
                new_sentence, new_sentence_tags = [], []

        dataset = []
        for sen, tags in zip(sentences, sentence_tags):
            numeric_tags = [0 if t == 'O' else 1 for t in tags]
            dataset.append([sen, numeric_tags])

    return dataset


def main():
    file_path = r"./data/train.tagged"
    windows_size = 0
    comp_parse_file(file_path, windows_size)


if __name__ == '__main__':
    main()
