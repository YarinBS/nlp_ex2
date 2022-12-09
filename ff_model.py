import torch
from torch import nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from parser import parse_file
from create_embedding import load_model, generate_ds
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, vec_dim, num_classes, hidden_dim=100):
        super().__init__()
        self.first_layer = nn.Linear(vec_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()
        self.out_func = nn.Sigmoid()

    def forward(self, x):
        x = self.first_layer(x)
        x = self.activation(x)
        x = self.second_layer(x)
        x = self.out_func(x)
        return x


# ---------------------
def train(model, train_ds, validation_ds, optimizer, num_epochs: int):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_ds, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, num_classes):
        self.x = x
        self.y = y
        self.num_classes = num_classes

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx]
        label = torch.zeros(self.num_classes)
        label[self.y[idx]] = 1
        return sample, label


def main():
    train_file_path = r'./data/train.tagged'
    validation_file_path = r'./data/dev.tagged'
    windows_size = 2
    glove_model = load_model()
    train_set = parse_file(file_path=train_file_path,
                           windows_size=windows_size)

    validation_set = parse_file(file_path=validation_file_path,
                                windows_size=windows_size)

    x_train, y_train = generate_ds(glove_model, train_set)
    x_validation, y_validation = generate_ds(glove_model, validation_set)

    train_ds = DataLoader(CustomDataset(x=x_train, y=y_train), batch_size=16, shuffle=True)
    validation_ds = DataLoader(CustomDataset(x=x_validation, y=y_validation), batch_size=16, shuffle=False)

    model = Net(vec_dim=(windows_size * 2 + 1) * 200, num_classes=2, hidden_dim=100)
    train(model=model,
          train_ds=train_ds,
          validation_ds=validation_ds,
          optimizer=torch.optim.Adam(model.parameters(), 1e-4),
          num_epochs=100)


if __name__ == '__main__':
    main()
