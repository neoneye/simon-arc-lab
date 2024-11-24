import torch
import random
import numpy as np
from transformer_classifier import TransformerClassifier

model_path = '/Users/neoneye/git/python_arc/run_tasks_result/transformer_classifier.pth'

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize and load the trained model
model = TransformerClassifier(src_vocab_size=528, num_classes=10)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

def pad_sequence(sequence, max_length, pad_value=0):
    if len(sequence) < max_length:
        sequence = sequence + [pad_value] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

def predict(model, input_sequence, device, max_length):
    # Pad the input sequence
    input_sequence = pad_sequence(input_sequence, max_length)
    
    # Convert input to tensor and add batch dimension
    src = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        output = model(src)
    
    # Get predicted class
    predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class

# Define the maximum sequence length used during training
max_length = 42

# Example input sequence
input_sequence = [234, 77, 1, 217, 203, 38, 127, 242, 288, 287, 257, 257,
                  361, 349, 296, 290, 328, 328, 400, 395, 287, 293, 288, 289,
                  441, 361, 360, 360, 360, 360, 360, 360, 360, 360, 360, 349,
                  349, 349, 359, 359, 359, 359]

# Perform prediction
predicted_class = predict(model, input_sequence, device, max_length)
print(f'Predicted class: {predicted_class}')
