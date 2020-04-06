import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm


class Data(Dataset):
    def __init__(self, dataset_dir, train=True):

        if train:
            data_path = os.path.join(dataset_dir, 'training')
        else:
            data_path = os.path.join(dataset_dir, 'testing')

        classes = os.listdir(data_path)
        self.data = []
        self.labels = []

        for label in classes:
            image_paths = os.listdir(os.path.join(data_path, label))
            self.data.extend([os.path.join(data_path, label, image) for image in image_paths])
            self.labels.extend([label for i in range(len(image_paths))])

        self.labels = torch.from_numpy(np.array(self.labels).astype(int)).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        trans_tensor = transforms.ToTensor()
        img = trans_tensor(Image.open(img))

        return img, label


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        # flatten after batch size [batch_size, channels, rows, columns] --> [channels, rows, columns]
        x = torch.flatten(x, start_dim=1)
        # print(x.size())
        x = self.fc1(x)
        output = self.fc2(x)
        return output


def main():
    # Variables
    batch_size = 32
    test_batch_size = 1000
    num_epochs = 100
    temp_loss = np.Inf
    # Load data
    train_data = Data(dataset_dir='data')
    val_data = Data(dataset_dir='data', train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=True)
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load Model
    model = Model()
    model.to(device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())  # default lr 0.001 used
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode='min', verbose=True)
    # Loss
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, num_epochs + 1):
        model.train()
        train_pbar = tqdm(train_loader)
        train_pbar.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
        train_loss = 0
        for batch_id, data in enumerate(train_pbar):
            input, target = data
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # print statistics
            train_loss += loss.item()
            if batch_id % 10 == 0:
                train_pbar.set_postfix(loss=train_loss/(batch_id + 1))

        # Run model over validation data
        model.eval()
        val_pbar = tqdm(val_loader)
        val_pbar.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
        val_loss = 0

        with torch.no_grad():
            for batch_id, data in enumerate(val_pbar):
                input, target = data
                input, target = input.to(device), target.to(device)
                output = model(input)
                loss = criterion(output, target)
                # print statistics
                val_loss += loss.item()
                val_pbar.set_postfix(val_loss=val_loss / (batch_id + 1))

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < temp_loss:
            temp_loss = val_loss
            torch.save(model.state_dict(), 'mnist.pth')


if __name__ == '__main__':
    main()