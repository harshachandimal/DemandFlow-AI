import torch
import torch.nn as nn
import joblib

class RossmannEnhancedFutureAwareLSTM(nn.Module):
    def __init__(self, past_features: int, future_features: int, forecast_days: int):
        super().__init__()
        self.forecast_days = forecast_days
        self.past_lstm = nn.LSTM(
            input_size=past_features, hidden_size=64, num_layers=2,
            batch_first=True, dropout=0.3
        )
        self.future_encoder = nn.Sequential(
            nn.Linear(future_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(64 + 32, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, x_past, x_future):
        _, (hn, _) = self.past_lstm(x_past)
        past_context  = hn[-1]
        past_repeated = past_context.unsqueeze(1).repeat(1, self.forecast_days, 1)
        future_enc    = self.future_encoder(x_future)
        fused         = torch.cat([past_repeated, future_enc], dim=2)
        return self.prediction_head(fused).squeeze(-1)


def load_model_and_scalers():
    # Number of features from v2
    past_feature_cols = 20
    future_feature_cols = 18
    forecast_days = 7

    model = RossmannEnhancedFutureAwareLSTM(
        past_features=past_feature_cols,
        future_features=future_feature_cols,
        forecast_days=forecast_days
    )
    model.load_state_dict(torch.load("models/real/rossmann_store_1_lstm_v2_model.pth", map_location="cpu"))
    model.eval()

    past_scaler = joblib.load("models/real/rossmann_v2_past_feature_scaler.pkl")
    future_scaler = joblib.load("models/real/rossmann_v2_future_feature_scaler.pkl")
    target_scaler = joblib.load("models/real/rossmann_v2_target_scaler.pkl")

    return model, past_scaler, future_scaler, target_scaler
