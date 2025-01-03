
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.seq2seq import Seq2Seq
from utils.data_processing import generate_data, save_data_to_csv
from train import train_model
from test import evaluate_model
import matplotlib.pyplot as plt
# Configurations
input_size = 10
output_size = 10
hidden_size = 256
num_layers = 2
num_epochs = 10
batch_size = 64
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate and save dataset
source_data, target_data = generate_data()
save_data_to_csv(source_data, target_data, 'data/dataset.csv')

# Train-test split
train_size = int(0.8 * len(source_data))
train_source, train_target = source_data[:train_size], target_data[:train_size]
test_source, test_target = source_data[train_size:], target_data[train_size:]

train_dataset = TensorDataset(train_source, train_target)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_source, test_target)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = Seq2Seq(input_size, output_size, hidden_size, num_layers).to(device)

# Train the model
train_loss = train_model(model, train_loader, num_epochs, learning_rate, device)

# Plot the training loss
plt.plot(train_loss)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluate the model
evaluate_model(model, test_loader, device)
