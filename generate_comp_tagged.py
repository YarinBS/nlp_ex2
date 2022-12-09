from ff_model import CustomDataset
from torch.utils.data import DataLoader
from create_embedding import generate_ds, load_model
from parser import parse_file
def main():
    validation_file_path = r'./data/dev.tagged'
    windows_size = 2
    glove_model = load_model()
    train_set = parse_file(file_path=train_file_path,
                           windows_size=windows_size)
    x_validation, y_validation = generate_ds(glove_model, validation_set)

    validation_ds = DataLoader(CustomDataset(x=x_validation, y=y_validation, num_classes=2), batch_size=16,
                               shuffle=False)

    with torch.no_grad():
        for idx, data in enumerate(validation_ds, 0):
            inputs, labels = data
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)

            outputs = model(inputs)
            validation_loss += criterion(outputs, labels).item()
            validation_f1.append(f1_metric(outputs, labels).item())
            




if __name__ == '__main__':
    main()