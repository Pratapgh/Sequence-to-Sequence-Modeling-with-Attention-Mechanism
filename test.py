print("Model Testing begins...")
import torch

def evaluate_model(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for source_batch, target_batch in test_loader:
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            outputs = model(source_batch, target_batch)

            # Predicted tokens
            predicted_tokens = outputs.argmax(dim=-1)  # [batch_size, seq_len]
            total_correct += (predicted_tokens[:, 1:] == target_batch[:, 1:]).sum().item()
            total_samples += target_batch[:, 1:].numel()

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
