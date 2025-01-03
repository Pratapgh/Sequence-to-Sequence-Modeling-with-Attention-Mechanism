
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    total_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for source_batch, target_batch in train_loader:
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)

            optimizer.zero_grad()
            output = model(source_batch, target_batch)

            # Reshape outputs and targets for loss calculation
            output = output[:, 1:].reshape(-1, model.decoder.fc_out.out_features)
            target_batch = target_batch[:, 1:].reshape(-1)

            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_avg_loss = epoch_loss / len(train_loader)
        total_loss.append(epoch_avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_avg_loss}")
    return total_loss

print("Training done")
