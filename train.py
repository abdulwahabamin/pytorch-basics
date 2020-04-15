import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn as nn
from tqdm import tqdm


class Data(Dataset):
    def __init__(self, data_dir, train=True):

        if train:
            data_path = os.path.join(data_dir, 'training')
        else:
            data_path = os.path.join(data_dir, 'testing')

        classes = os.listdir(data_path)  # list classes

        self.data = []
        self.labels = []

        for label in classes:
            images = os.listdir(os.path.join(data_path, label))  # get images
            self.data.extend([os.path.join(data_path, label, image) for image in images])  # add image paths to list
            self.labels.extend([label for i in range(len(images))])  # add corresponding label

    def __len__(self):
        return len(self.data)  # return total number of examples

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        trans_tensor = transforms.ToTensor()
        img = trans_tensor(Image.open(img))  # load image and convert to tensor
        label = torch.from_numpy(np.asarray(label).astype(int)).long()  # convert string label to long datatype

        return img, label


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)  # input is [batch_size, channels, rows, columns]
        x = self.fc1(x)
        output = self.fc2(x)

        return output


def main():
    # Variables
    data_dir = 'data'
    batch_size = 32
    val_batch_size = 1000
    num_epochs = 100
    temp_loss = np.Inf
    # Load data
    train_data = Data(data_dir=data_dir)
    val_data = Data(data_dir=data_dir, train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=True)
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load Model
    model = Model()
    model.to(device)  # transfer model parameters to available device
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode='min', verbose=True)
    # Loss
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, num_epochs +1):
        model.train()
        train_pbar = tqdm(train_loader)
        train_pbar.set_description('Epoch {}/{}'.format(epoch +1, num_epochs))
        train_loss = 0

        for batch_id, data in enumerate(train_pbar):
            input, target = data
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_id % 10 == 0:
                train_pbar.set_postfix(loss=train_loss / (batch_id + 1))

        # Performing Validation
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
                val_loss += loss.item()
                if batch_id % 10 == 0:
                    val_pbar.set_postfix(val_loss=val_loss / (batch_id + 1))

        val_loss = val_loss / (len(val_loader))
        scheduler.step(val_loss)  # using validation

        if val_loss < temp_loss:
            temp_loss = val_loss
            torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()