import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from create_embedding import generate_ds, load_model
from parser import comp_parse_file
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score
import numpy as np


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, num_classes):
        self.x = x
        self.y = y
        self.num_classes = num_classes

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx]
        label = torch.zeros(len(self.y[idx]), self.num_classes)
        for idx, l in enumerate(self.y[idx]):
            label[idx][l] = 1

        return sample, label


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=200, hidden_size=2, num_layers=2, bidirectional=False, dropout=0.1)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.out_func = nn.Softmax(dim=-1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        tag_scores = self.out_func(lstm_out)
        # tag_scores = F.log_softmax(lstm_out, dim=-1)
        return tag_scores


def main():
    # lstm = nn.LSTM(200, 2)
    # hidden = (torch.randn(1, 1, 200),
    #           torch.randn(1, 1, 200),
    #           torch.randn(1, 1, 200))
    # inputs = [torch.randn(1, 200) for _ in range(5)]
    #
    # inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    # out, hidden = lstm(inputs)
    #
    # inputs = [torch.randn(1, 200) for _ in range(12)]
    #
    # inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    #
    # lstm(inputs)
    # print('hi')

    train_file_path = r'./data/train.tagged'
    validation_file_path = r'./data/dev.tagged'
    windows_size = 2
    glove_model = load_model()
    train_set = comp_parse_file(file_path=train_file_path,
                                windows_size=windows_size)

    validation_set = comp_parse_file(file_path=validation_file_path,
                                     windows_size=windows_size)

    x_train, y_train = generate_ds(glove_model, train_set, comp=True)
    x_validation, y_validation = generate_ds(glove_model, validation_set, comp=True)

    train_ds = DataLoader(CustomDataset(x=x_train, y=y_train, num_classes=2), batch_size=1, shuffle=True)
    validation_ds = DataLoader(CustomDataset(x=x_validation, y=y_validation, num_classes=2), batch_size=1,
                               shuffle=False)

    criterion = nn.CrossEntropyLoss()
    f1_metric = BinaryF1Score()

    model = LSTMTagger(embedding_dim=200, hidden_dim=5, vocab_size=23232, tagset_size=2)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    loss_lst = []
    for epoch in range(50):  # loop over the dataset multiple times
        train_f1 = []
        for i, data in enumerate(train_ds, 0):
            inputs, labels = data
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.

            # Step 3. Run our forward pass.
            tag_scores = model(inputs)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            total_loss = None

            for c, l in zip(tag_scores[0], labels[0]):
                if total_loss is None:
                    total_loss = criterion(c, l)
                else:
                    total_loss += criterion(c, l)

                train_f1.append(f1_metric(c, l).item())

            loss_lst.append(total_loss.item())
            total_loss.backward()
            optimizer.step()

        print(f'[{epoch + 1}, train loss: {np.mean(loss_lst):.3f}, f1: {np.mean(train_f1)}')

        loss_lst = []
        train_f1 = []

        validation_f1 = []
        validation_loss = []
        with torch.no_grad():
            for idx, data in enumerate(validation_ds, 0):
                inputs, labels = data
                inputs = inputs.to(torch.float32)
                labels = labels.to(torch.float32)

                outputs = model(inputs)

                for c, l in zip(outputs[0], labels[0]):
                    validation_loss.append(criterion(c, l).item())
                    validation_f1.append(f1_metric(c, l).item())

        print(f'[{epoch + 1}, validation loss: {np.mean(validation_loss) / idx:.3f} f1: {np.mean(validation_f1)}')


if __name__ == '__main__':
    main()
