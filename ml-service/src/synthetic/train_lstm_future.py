import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import matplotlib.pyplot as plt

# 6. Build model class: FutureAwareLSTM
class FutureAwareLSTM(nn.Module):
    def __init__(self, past_input_size=6, future_input_size=4, hidden_size=64, num_layers=2, dropout=0.2, forecast_days=7):
        super(FutureAwareLSTM, self).__init__()
        self.forecast_days = forecast_days
        
        # A. Past encoder
        # This LSTM processes the past 30 days of data
        self.past_encoder = nn.LSTM(
            input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # B. Future feature encoder
        # This processes the known future features for each of the 7 forecast days independently
        self.future_encoder_linear = nn.Linear(future_input_size, 16)
        self.future_encoder_relu = nn.ReLU()
        
        # D. Prediction head
        # This combines the encoded past context with the encoded future features
        # Past context size (64) + Future encoded size (16) = 80
        self.head_linear1 = nn.Linear(hidden_size + 16, 32)
        self.head_relu = nn.ReLU()
        self.head_linear2 = nn.Linear(32, 1)

    def forward(self, x_past, x_future):
        # x_past shape: (batch_size, 30, past_input_size)
        # x_future shape: (batch_size, 7, future_input_size)
        
        # Process past sequence
        lstm_out, _ = self.past_encoder(x_past)
        
        # Extract the final hidden state from the LSTM to serve as our "past context"
        # Shape: (batch_size, hidden_size)
        past_context = lstm_out[:, -1, :]
        
        # C. Fusion
        # Repeat the past context 7 times, once for each forecast day
        # Shape: (batch_size, 7, hidden_size)
        past_context_repeated = past_context.unsqueeze(1).repeat(1, self.forecast_days, 1)
        
        # Process future features
        # Shape becomes: (batch_size, 7, 16)
        encoded_future = self.future_encoder_relu(self.future_encoder_linear(x_future))
        
        # Concatenate repeated past context with encoded future features
        # Shape: (batch_size, 7, hidden_size + 16)
        combined = torch.cat((past_context_repeated, encoded_future), dim=2)
        
        # D. Prediction head
        # Process the combined features through the prediction head
        out = self.head_relu(self.head_linear1(combined))
        out = self.head_linear2(out) # Shape: (batch_size, 7, 1)
        
        # Remove the last dimension to match target shape: (batch_size, 7)
        return out.squeeze(2)

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean(np.square(y_pred - y_true)))
    mape = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)) * 100
    return mae, rmse, mape

def main():
    # 1. Load arrays
    data_path = 'data/processed/synthetic_future_sequences.npz'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print("Loading dual-input sequences...")
    with np.load(data_path) as data:
        X_past_train_np = data['X_past_train']
        X_future_train_np = data['X_future_train']
        y_train_np = data['y_train']
        X_past_test_np = data['X_past_test']
        X_future_test_np = data['X_future_test']
        y_test_np = data['y_test']

    # 12. Print shapes
    print("\n--- Data Shapes ---")
    print(f"X_past_train shape: {X_past_train_np.shape}")
    print(f"X_future_train shape: {X_future_train_np.shape}")
    print(f"y_train shape: {y_train_np.shape}")
    print(f"X_past_test shape: {X_past_test_np.shape}")
    print(f"X_future_test shape: {X_future_test_np.shape}")
    print(f"y_test shape: {y_test_np.shape}")

    # 2. Load target scaler
    scaler_path = 'models/future_target_scaler.pkl'
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found.")
        return
    target_scaler = joblib.load(scaler_path)

    # 3. Convert arrays to PyTorch tensors
    X_past_train = torch.tensor(X_past_train_np, dtype=torch.float32)
    X_future_train = torch.tensor(X_future_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    
    X_past_test = torch.tensor(X_past_test_np, dtype=torch.float32)
    X_future_test = torch.tensor(X_future_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    # 4. Create TensorDataset
    train_dataset = TensorDataset(X_past_train, X_future_train, y_train)
    test_dataset = TensorDataset(X_past_test, X_future_test, y_test)

    # 5. Create DataLoader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    past_input_size = X_past_train_np.shape[2]
    future_input_size = X_future_train_np.shape[2]
    forecast_days = y_train_np.shape[1]
    
    model = FutureAwareLSTM(
        past_input_size=past_input_size,
        future_input_size=future_input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        forecast_days=forecast_days
    )
    
    print("\n--- Model Architecture ---")
    print(model)

    # 7. Use MSELoss, Adam optimizer, lr=0.001, epochs=120
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 120

    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch_x_past, batch_x_future, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x_past, batch_x_future)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x_past.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 8. Print training loss every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.6f}")

    print("Training complete.")

    # 9. Evaluate on test set
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch_x_past, batch_x_future, batch_y in test_loader:
            outputs = model(batch_x_past, batch_x_future)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x_past.size(0)
            
            all_predictions.append(outputs.numpy())
            all_actuals.append(batch_y.numpy())

    test_loss /= len(test_loader.dataset)
    
    # 12. Print final test MSE
    print(f"\nFinal Test Loss (MSE): {test_loss:.6f}")

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)

    # 10. Inverse transform y_test and predictions
    y_test_real = target_scaler.inverse_transform(all_actuals.reshape(-1, 1)).reshape(-1, forecast_days)
    predictions_real = target_scaler.inverse_transform(all_predictions.reshape(-1, 1)).reshape(-1, forecast_days)

    # 11. Calculate metrics
    mae, rmse, mape = calculate_metrics(y_test_real, predictions_real)
    
    # 12. Print MAE, RMSE, MAPE
    print("\n--- Overall Test Metrics (Future-Aware Model) ---")
    print(f"MAE:  {mae:.2f} units")
    print(f"RMSE: {rmse:.2f} units")
    print(f"MAPE: {mape:.2f}%")

    # 12. Print one actual next 7 days sample and one predicted next 7 days sample
    sample_idx = 0
    print("\n--- Prediction vs Actual on First Test Sample ---")
    print("Actual Next 7 Days (Units Sold):   ", np.round(y_test_real[sample_idx], 1))
    print("Predicted Next 7 Days (Units Sold):", np.round(predictions_real[sample_idx], 1))

    # 13. Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/lstm_future_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # 15. Save metrics to text file
    os.makedirs('reports', exist_ok=True)
    metrics_path = 'reports/lstm_future_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Test MSE: {test_loss:.6f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAPE: {mape:.2f}\n")
    print(f"Metrics saved to {metrics_path}")

    # 14. Save chart
    plt.figure(figsize=(8, 5))
    days = range(1, forecast_days + 1)
    
    plt.plot(days, y_test_real[sample_idx], marker='o', label='Actual Sales', color='blue')
    plt.plot(days, predictions_real[sample_idx], marker='x', label='Predicted Sales', color='green', linestyle='--')
    
    plt.title('Future-Aware LSTM: Prediction vs Actual Sales (Next 7 Days)')
    plt.xlabel('Day in Forecast Window')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    chart_path = 'reports/lstm_future_prediction_vs_actual.png'
    plt.savefig(chart_path)
    plt.close()
    print(f"Chart saved to {chart_path}")

if __name__ == "__main__":
    main()
