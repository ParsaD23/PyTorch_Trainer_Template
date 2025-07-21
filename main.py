from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from trainer import Trainer

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def output_parse(self, output): # modify based on desired output
        return torch.max(output, 1)[1]

class IrisDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input": self.inputs[idx],
            "target": self.targets[idx]
        }

def main():
    random_state = 42

    # Load and preprocess
    data = load_iris()
    X = data.data
    y = data.target

    # Split into train, val, test (each 50%, then 50/50 again)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create datasets and dataloaders
    train_dataset = IrisDataset(X_train_tensor, y_train_tensor)
    val_dataset = IrisDataset(X_val_tensor, y_val_tensor)
    test_dataset = IrisDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(model=model, device='cpu')

    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=100,
        early_stopping=True,
        early_stopping_monitor='accuracy',
        early_stopping_mode='max',
        metrics={
            'accuracy': accuracy_score
        },
        fast_dev_run=False,
    )

    print(trainer.test(test_loader, criterion, metrics={
            'accuracy': accuracy_score
        }))
    print(trainer.predict(next(iter(test_loader))['input'][0]))

if __name__=='__main__':
    main()
