# Sequence-to-Sequence-Modeling-with-Attention-Mechanism
A Sequence-to-Sequence (Seq2Seq) model with attention mechanism for sequence transformation tasks. The project includes data generation, training, and evaluation scripts, utilizing PyTorch for implementation. Features modular code structure with CSV-based dataset handling and visualization of training loss curves.


**Seq2Seq Model with Attention Mechanism**

This project implements a Sequence-to-Sequence (Seq2Seq) model with an attention mechanism for sequence transformation tasks using PyTorch. The model is trained on synthetic data and evaluates performance in reversing sequences.

**Features**

**.Seq2Seq Architecture** : Includes Encoder, Attention, and Decoder components.

**.Attention Mechanism** : Enables better handling of long input sequences by focusing on relevant parts.

**.Synthetic Dataset** : Automatically generates source and target sequences for training.

**.Training and Testing** : Includes scripts for training the model and evaluating accuracy.

**.Visualization** : Plots the training loss curve for monitoring progress.

**.Modular Design** : Organized with separate files and folders for models, utilities, and data.


**Project Structure**

```
Seq2Seq_Model_Project/
│
├── data/                     # Dataset-related files
│   └── dataset.csv           # Source and target sequences
│
├── models/                   # Model definitions
│   ├── encoder.py            # Encoder class
│   ├── attention.py          # Attention class
│   ├── decoder.py            # Decoder class
│   └── seq2seq.py            # Seq2Seq model
│
├── utils/                    # Utility functions
│   └── data_processing.py    # Data generation and processing
│
├── train.py                  # Training script
├── test.py                   # Testing script
├── main.py                   # Script to execute the full workflow
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── notebooks/                # (Optional) Experimentation notebooks
    └── seq2seq_model.ipynb
```

## Results
### Training Loss
The training loss decreased over 10 epochs, showing effective learning:
- **Epoch 1 Loss**:  2.16299
- **Epoch 5 Loss**:  1.17597
- **Epoch 10 Loss**: 0.26381  

### Test Accuracy
The model achieved **94.33% accuracy** on the test dataset.

## Prerequisites
- Python 3.9 or higher
- CUDA-enabled GPU (optional for faster training)

## Installation
1. Clone the repository:  
   ```bash
   git clone <repository-url>
   cd Seq2Seq_Model_Project
   ```
2. Set up a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Generate Dataset**:  
   Run the data processing script to create `dataset.csv` in the `data/` folder:  
   ```bash
   python utils/data_processing.py
   ```
2. **Train the Model**:  
   Train the Seq2Seq model using `train.py`:  
   ```bash
   python train.py
   ```
3. **Test the Model**:  
   Evaluate accuracy on the test dataset:  
   ```bash
   python test.py
4. **Get the results**:  
   Get accuracy and loss curve on the test dataset:  
   ```bash
   python main.py
   ```

## Dependencies
- PyTorch  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  

For more details, see `requirements.txt`.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

---

