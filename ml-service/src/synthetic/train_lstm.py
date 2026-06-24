import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import matplotlib.pyplot as plt

# 5. Build LSTM model class called DemandLSTM
class DemandLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, forecast_days=7):
        super(DemandLSTM, self).__init__()
        
        # LSTM layer expects input shape: (batch_size, sequence_length, input_size) if batch_first=True
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # We project the LSTM's final hidden state to a smaller intermediate space
        self.linear1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        # Finally project to our output shape (forecast_days)
        self.linear2 = nn.Linear(32, forecast_days)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # out shape: (batch_size, sequence_length, hidden_size)
        # We only care about the very last hidden state in the sequence for forecasting
        out, _ = self.lstm(x)
        
        # Extract the hidden state of the last time step
        last_out = out[:, -1, :]  # shape: (batch_size, hidden_size)
        
        # Pass through linear layers
        last_out = self.linear1(last_out)
        last_out = self.relu(last_out)
        predictions = self.linear2(last_out) # shape: (batch_size, forecast_days)
        
        return predictions

def main():
    # 1. Load processed arrays
    data_path = 'data/processed/synthetic_sequences.npz'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print("Loading sequences...")
    with np.load(data_path) as data:
        # Check keys, but we know them from preprocessing script
        X_train_np = data['X_train']
        y_train_np = data['y_train']
        X_test_np = data['X_test']
        y_test_np = data['y_test']

    # 11. Print shapes
    print("\n--- Data Shapes ---")
    print(f"X_train shape: {X_train_np.shape}")
    print(f"y_train shape: {y_train_np.shape}")
    print(f"X_test shape: {X_test_np.shape}")
    print(f"y_test shape: {y_test_np.shape}")

    # 2. Load target scaler
    scaler_path = 'models/target_scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found.")
        return
        
    target_scaler = joblib.load(scaler_path)

    # 3. Convert arrays to PyTorch tensors
    # Machine learning frameworks like PyTorch use tensors (which are like multi-dimensional arrays optimized for GPU)
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    # 4. Create TensorDataset and DataLoader
    # DataLoader handles batching and shuffling automatically
    # Important: We only shuffle batches inside the train loader (after chronological split)
    # to prevent the model from learning the exact sequence of batches, which improves generalization.
    batch_size = 16
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = X_train_np.shape[2] # Number of features
    forecast_days = y_train_np.shape[1] # Target prediction length
    
    model = DemandLSTM(
        input_size=input_size, 
        hidden_size=64, 
        num_layers=2, 
        dropout=0.2, 
        forecast_days=forecast_days
    )
    
    print("\n--- Model Architecture ---")
    print(model)

    # 6. Loss function, optimizer, settings
    # MSELoss computes the Mean Squared Error between predictions and actuals
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 120

    print(f"\nStarting training for {epochs} epochs...")
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad() # Clear previous gradients
            
            # Forward pass: compute predictions
            outputs = model(batch_X)
            
            # Compute loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 7. Print loss every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.6f}")

    print("Training complete.")

    # 8. Evaluate test loss
    model.eval() # Set model to evaluation mode (turns off dropout, etc.)
    test_loss = 0.0
    all_predictions = []
    all_actuals = []

    with torch.no_grad(): # Don't track gradients during evaluation
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
            
            all_predictions.append(outputs.numpy())
            all_actuals.append(batch_y.numpy())

    test_loss /= len(test_loader.dataset)
    
    # Print final test loss
    print(f"\nFinal Test Loss (MSE): {test_loss:.6f}")

    # Combine batches back into full arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)

    # Grab the very first sample from our test set for visualization
    sample_idx = 0
    sample_pred_scaled = all_predictions[sample_idx]
    sample_actual_scaled = all_actuals[sample_idx]

    # Inverse transform predictions back to real units_sold
    # The scaler expects a 2D array, so we reshape
    sample_pred_real = target_scaler.inverse_transform(sample_pred_scaled.reshape(-1, 1)).flatten()
    sample_actual_real = target_scaler.inverse_transform(sample_actual_scaled.reshape(-1, 1)).flatten()

    print("\n--- Prediction vs Actual on First Test Sample ---")
    print("Actual Next 7 Days (Units Sold):", np.round(sample_actual_real, 1))
    print("Predicted Next 7 Days (Units Sold):", np.round(sample_pred_real, 1))

    # 9. Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/lstm_demand_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # 10. Save chart comparing actual vs predicted
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(8, 5))
    days = range(1, forecast_days + 1)
    
    plt.plot(days, sample_actual_real, marker='o', label='Actual Sales', color='blue')
    plt.plot(days, sample_pred_real, marker='x', label='Predicted Sales', color='red', linestyle='--')
    
    plt.title('LSTM Prediction vs Actual Sales (Next 7 Days)')
    plt.xlabel('Day in Forecast Window')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    chart_path = 'reports/lstm_prediction_vs_actual.png'
    plt.savefig(chart_path)
    plt.close()
    print(f"Chart saved to {chart_path}")

if __name__ == "__main__":
    main()
