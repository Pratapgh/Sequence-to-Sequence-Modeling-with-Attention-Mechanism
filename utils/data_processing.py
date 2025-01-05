
import numpy as np
import pandas as pd
import torch

def generate_data(num_samples=1000, seq_len=10, input_size=10):
    source_seq = np.random.randint(1, input_size, size=(num_samples, seq_len))
    target_seq = np.flip(source_seq, axis=1)  # Target is the reverse of source
    return torch.tensor(source_seq.copy(), dtype=torch.long), torch.tensor(target_seq.copy(), dtype=torch.long)

def save_data_to_csv(source_data, target_data, filepath):
    data_dict = {
        'source': [','.join(map(str, seq)) for seq in source_data.numpy()],
        'target': [','.join(map(str, seq)) for seq in target_data.numpy()]
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(filepath, index=False)
