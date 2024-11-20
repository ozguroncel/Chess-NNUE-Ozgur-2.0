import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import feature_encoding_1144


# Step 1: Load Data and Convert FENs to Feature Vectors

def load_data(*file_paths):
    fens = []
    evals = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                if 'Evaluation' in line:
                    fen, eval_score = line.split('|')
                    eval_score = int(eval_score.replace("Evaluation:", "").strip())
                    fens.append(fen.strip())
                    evals.append(normalize_evaluation(eval_score))

    # Convert FENs to input feature vectors
    X = np.array([feature_encoding_1144.fen_to_detailed_feature_vector(fen) for fen in fens])
    y = np.array(evals)

    return X, y

def normalize_evaluation(eval_score):
    # Cap extreme values to handle checkmate evaluations
    if eval_score > 5000:
        eval_score = 5000
    elif eval_score < -5000:
        eval_score = -5000

    # Normalize to range [-1, 1]
    return eval_score / 5000

# Load the dataset
X, y = load_data("evaluated_positions/lichess_evaluations.txt", "evaluated_positions/chesscom_evaluations.txt")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  # Unsqueeze to match output size
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Step 2: Define Model, Loss, and Optimizer

model = feature_encoding_1144.CustomNNUE(input_size=1144).to(device)  # Adjusted input size to match feature count
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate for stability
loss_fn = nn.SmoothL1Loss()  # Huber Loss to handle outliers

# Step 3: Define Training Loop

def train_model(model, X_train, y_train, X_test, y_test, num_epochs=20, batch_size=32):
    """Train the model and evaluate test loss at each epoch."""
    model.train()
    
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        permutation = torch.randperm(X_train.size()[0])

        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            # Forward pass
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_loss / len(X_train)
        train_losses.append(avg_train_loss)

        # Evaluate test loss after each epoch
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = loss_fn(test_outputs, y_test).item()
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}")

    return train_losses, test_losses

# Step 4: Model training

train_losses, test_losses = train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_epochs=25, batch_size=32)

# Step 5: Plot the Losses

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the losses
plot_losses(train_losses, test_losses)

# Save the model at the end of training
torch.save(model.state_dict(), "3rd_model.pth")  # For the 3rd model, refers to the 1144 version.
