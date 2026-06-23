# DemandFlow AI - ML Service

This repository contains the Machine Learning service for DemandFlow AI, an advanced RNN/LSTM-based business demand forecasting project.

## Phase 1.1: Synthetic Data Generation

### What this dataset represents
The generated dataset (`data/sample_sales.csv`) represents 365 days of realistic daily sales data for a single product ("Rice 5kg"). It simulates common retail patterns such as baseline demand, weekend spikes, seasonal variations, marketing promotions, a gradual growth trend, and random day-to-day fluctuations.

### What each column means
- **`date`**: The specific day of the recorded sales (Format: YYYY-MM-DD).
- **`product`**: The name of the item sold ("Rice 5kg").
- **`units_sold`**: The target variable we eventually want to predict. It's the total number of items sold on that date. It is calculated by combining base demand, boosts, trend, and noise.
- **`price`**: The selling price of the product on that date. Prices may drop during promotions.
- **`stock`**: The inventory level at the end of the day.
- **`promotion`**: A boolean/binary indicator (1 = Yes, 0 = No) showing if a marketing promotion was active.
- **`day_of_week`**: An integer representing the day of the week (0 = Monday, 6 = Sunday). This helps the ML model capture weekly seasonality.
- **`month`**: An integer representing the month (1 = January, 12 = December). This helps the ML model capture yearly seasonality.

### Why time-series order matters
Unlike traditional tabular datasets where the order of rows doesn't matter (e.g., predicting house prices based on features), in time-series forecasting, **sequence is critical**.

Models like LSTMs (Long Short-Term Memory networks) rely on the chronological order of data to understand patterns across time. The sequence of past events (e.g., sales over the last 7 days) is directly used to predict future events (sales tomorrow). If the rows were shuffled, the model would lose the context of time, making it impossible to learn trends, seasonality, or temporal dependencies. Therefore, the data must remain sorted by `date`.

### How to run the script

1. Make sure you have Python and `pandas` installed:
   ```bash
   pip install pandas numpy
   ```

2. Navigate to the `ml-service` directory:
   ```bash
   cd ml-service
   ```

3. Run the generation script:
   ```bash
   python src/generate_sample_data.py
   ```

The script will generate the CSV file at `data/sample_sales.csv` and print out summary statistics.

## Phase 1.1.2: Exploratory Data Analysis (EDA)

### What EDA means
Exploratory Data Analysis (EDA) is the process of examining, summarizing, and visualizing a dataset to discover patterns, anomalies, and relationships between variables.

### Why we inspect data before training
Before feeding data into a complex model like an LSTM, we must understand its shape and quality. Inspecting data allows us to:
- Identify missing or corrupted values that would break the training process.
- Confirm that our predictive features (like `day_of_week` or `promotion`) actually have a visible relationship with our target (`units_sold`).
- Understand the scale of the target variable to decide on normalization strategies.
- Spot obvious trends or seasonalities that our model will be expected to learn.

### What each chart shows
- **`reports/sales_over_time.png`**: A line chart visualizing the entire 365-day sequence. It reveals the overall growth trend, daily noise, and broader seasonal waves.
- **`reports/sales_by_day_of_week.png`**: A bar chart demonstrating weekly seasonality, identifying which days consistently have higher or lower sales.
- **`reports/promotion_effect.png`**: A bar chart quantifying the impact of marketing events, proving whether the `promotion` feature is a strong signal for higher sales.
- **`reports/monthly_sales_pattern.png`**: A bar chart summarizing sales across different months, highlighting long-term seasonal effects.

### How to run the script

1. Make sure the sample data is already generated (`data/sample_sales.csv`).
2. Run the analysis script from the `ml-service` directory:
   ```bash
   python src/analyze_sample_data.py
   ```
3. Check the console output for statistical summaries and the `reports/` folder for the generated visualizations.

## Phase 1.2: Data Preprocessing & Sequence Creation

### What normalization means
Normalization (or scaling) transforms all numerical features to be on a similar scale, typically between 0 and 1. This helps neural networks like LSTMs train faster and more stably, as large values won't overpower small ones. We scale the features and the target separately so we can easily un-scale predictions later.

### Why LSTM needs sequences
LSTMs (Long Short-Term Memory networks) expect data in sequences because they are designed to learn from the progression of time. Instead of looking at a single day's data independently, an LSTM looks at a sequence of past days to understand the trajectory and context before making a prediction.

### What sliding windows are
To train the LSTM, we convert our continuous timeline into many smaller examples using "sliding windows". 
- An input window (lookback) contains a set number of consecutive days of features (e.g., 30 days).
- A target window (forecast) contains the following days of sales to predict (e.g., 7 days).
We slide this window one day forward at a time to generate thousands of overlapping examples from a single timeline.

### Why we use chronological split
In traditional machine learning, we randomly shuffle data before splitting it into train and test sets. In time-series forecasting, we **must split chronologically** (e.g., the first 80% of days for training, the last 20% for testing). Random shuffling would mix future data into the training set, causing "look-ahead bias" where the model unfairly learns from the future to predict the past.

### How to run the preprocessing script

1. Make sure you have the required libraries installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the preprocessing script from the `ml-service` directory:
   ```bash
   python src/preprocess_sequences.py
   ```
3. Check the console output for tensor shapes and look for the saved files in `data/processed/` and `models/`.

## Phase 1.3: Train LSTM Model

### What LSTM is doing
The LSTM is learning to map our historical sliding windows (e.g., 30 days of data) to a future sequence (e.g., 7 days of sales). By iterating over the dataset many times (epochs) and continuously adjusting its internal weights (using backpropagation and an optimizer like Adam), it discovers patterns linking the input features to the target output.

### What input shape means
Our input shape is `(batch_size, sequence_length, features)` (e.g., `16, 30, 6`). 
- `batch_size`: We process 16 sliding windows at a time.
- `sequence_length`: Each window looks back 30 days.
- `features`: Each day has 6 features (units sold, price, stock, promotion, etc.).

### What output shape means
Our output shape is `(batch_size, forecast_days)` (e.g., `16, 7`). For each of the 16 windows in a batch, the model outputs 7 continuous numerical values representing the predicted sales for the next 7 days.

### What MSE loss means
Mean Squared Error (MSE) measures how far off our predictions are from the actual values. It squares the differences (penalizing larger errors more heavily) and averages them. Our goal during training is to minimize this loss.

### How to run the training script
1. Ensure all requirements (including PyTorch) are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```bash
   python src/train_lstm.py
   ```
3. The script will train the model, output loss metrics, and generate an evaluation chart in `reports/`.

## Phase 1.4: Model Evaluation

### Why one prediction sample is not enough
During training, we visualized a single sequence to see if the model was learning. However, a model might just get lucky on one sample. To truly understand its performance, we must evaluate it across *all* unseen test samples to calculate aggregate error metrics.

### What MAE means
**Mean Absolute Error (MAE)** calculates the average absolute difference between predicted and actual sales. It tells us, on average, how many units we are off by (e.g., "we are off by 15 units on average"). It's highly intuitive for business stakeholders.

### What RMSE means
**Root Mean Squared Error (RMSE)** squares the errors before averaging them, and then takes the square root. Because it squares the errors, RMSE heavily penalizes large mistakes. If predicting 100 units off is considered much worse than being 10 units off 10 times, RMSE is the metric to watch.

### What MAPE means
**Mean Absolute Percentage Error (MAPE)** expresses the error as a percentage of the actual value. This is useful for understanding relative accuracy (e.g., "our predictions are off by 5%"). However, it can be unstable if actual sales are close to zero.

### Why business forecasting needs evaluation on all test samples
In business, supply chain and inventory decisions require knowing the average error margins. If the MAE is 5 units, a store might choose to stock 5 extra units as safety stock. Evaluating on the entire test set ensures these estimates are robust and reliable over varying seasons and conditions.

### How to run the evaluation script
```bash
python src/evaluate_lstm.py
```
Check `reports/` for detailed error analysis charts and a CSV of all predictions.

## Phase 1.5: Baseline Model Comparison

### What a baseline model is
A baseline model is a very simple, naive heuristic or rule-of-thumb method used for prediction, without any machine learning involved.

### Why neural networks must be compared against simple baselines
Machine learning models like LSTMs are complex and prone to overfitting or learning incorrect signals. If a naive rule-of-thumb can predict sales more accurately than a neural network, the neural network is completely useless. Proving the LSTM outperforms simple baselines is essential to justify its complexity and computational cost.

### What Last Value Baseline means
The **Last Value Baseline** simply predicts that future sales will be exactly the same as the sales on the very last known day. For example, if we sold 50 units today, we predict we will sell 50 units every day for the next 7 days. This is a common baseline for stable, slow-moving time series.

### What Weekly Seasonal Baseline means
The **Weekly Seasonal Baseline** predicts that future sales will exactly match the sales from the previous week. For example, if we sold 120 units last Monday, we predict we will sell 120 units next Monday. This is a strong baseline for retail data, which often has heavy weekly seasonality.

### How to interpret the comparison
By looking at the MAE, RMSE, and MAPE of all three models side-by-side, we can clearly see the "lift" (improvement) the LSTM provides over simple naive guesswork. A successful LSTM should significantly lower the error metrics compared to both baselines.

### How to run the baseline comparison
```bash
python src/compare_baselines.py
```
Check `reports/` for the `baseline_comparison.csv` and the `baseline_comparison_mape.png` chart.

## Phase 1.6: Future-Aware Sequence Preprocessing

### What future known covariates are
In real-world retail, we don't just know what happened in the past; we also know *some* things about the future. For example, we know tomorrow is a Tuesday (`day_of_week`), we know it's November (`month`), and we know if we have a marketing `promotion` scheduled. We may also know the planned selling `price`. These are called "known future covariates."

### Why future units_sold must not be included in X_future
While we can feed the model tomorrow's expected price and promotion, we absolutely cannot feed it tomorrow's `units_sold`. That is the exact variable the model is trying to predict! Including the target variable in the input for the same time step is called "data leakage" and invalidates the entire model. 

### Why this is more realistic than the first LSTM preprocessing
The first LSTM only looked at the past 30 days to predict the next 7 days. It essentially flew blind into the future. It couldn't see if a massive promotion was scheduled for Day 3 of its forecast window. By splitting our data into `X_past` (what happened) and `X_future` (what is planned), we provide the neural network with the exact same information a human planner would have, making the model significantly more realistic and powerful.

### How to run the preprocessing script
```bash
python src/preprocess_future_sequences.py
```
Check `data/processed/` for the new `synthetic_future_sequences.npz` file containing the dual-input arrays.

## Phase 1.7: Train Future-Aware LSTM

### Why this model uses X_past and X_future
Demand forecasting in the real world isn't just about looking backwards. A business actively plans future promotions, sets prices, and knows the upcoming days of the week. By designing a neural network that explicitly accepts an `X_past` input (historical context) and an `X_future` input (upcoming known context), the model can mathematically fuse "what usually happens" with "what is scheduled to happen."

### What future known features are
These are variables whose values are definitively known ahead of time. In our synthetic data, this includes `price`, `promotion`, `day_of_week`, and `month`.

### Why future units_sold is not used as input
If we fed tomorrow's `units_sold` into the network while it was trying to predict tomorrow's `units_sold`, the model would just learn to copy the input to the output. It would perform perfectly during training and fail completely in reality (since we don't know tomorrow's sales in real life). This is called Data Leakage.

### How this model is different from the first LSTM
The first LSTM squeezed all features into a single 30-day lookback window and blindly tried to project 7 days forward. The `FutureAwareLSTM` uses two pathways:
1. An LSTM encoder to process the 30-day history into a single "context" vector.
2. A Linear encoder to process the 7-day scheduled future features.
It then fuses the historical context with the future schedule to make highly informed predictions. This architecture allows it to easily react to sudden events like planned promotions.

### How to run the script
```bash
python src/train_lstm_future.py
```
Check `reports/` for the prediction chart and the saved metrics comparing its performance to the v1 model.

## Phase 1.8: Preprocessing v3 with Seasonal Lags

### What lag features are
Lag features are historical values of the target variable shifted forward in time. Instead of relying solely on the neural network to "remember" what happened 7 days ago, we explicitly create a column that hands the model that exact historical value. This drastically reduces the cognitive load on the LSTM.

### What same_day_last_week_sales means
If today is a Monday, `same_day_last_week_sales` is exactly the number of units sold last Monday. Because this value occurred 7 days ago, it is already "known" history, which means it is perfectly legal to provide it as an input for today's forecast without causing data leakage. 

### Why this helps beat the weekly seasonal baseline
In Phase 1.5, we saw that simply copying last week's sales (the Weekly Seasonal Baseline) outperformed our LSTM. The LSTM was struggling to mathematically extract that precise repeating pattern. By feeding `same_day_last_week_sales` directly into the network as a feature, we give the LSTM the baseline answer as a starting point. The LSTM can then focus on *adjusting* that baseline based on other features (like promotions and price drops), effectively combining the strengths of the heuristic baseline with the pattern recognition of deep learning.

### Why shift/rolling must not use future values
When calculating a rolling average, if we include today's sales to predict today's sales, that is data leakage. We must use a `shift(1)` to ensure the 7-day rolling average only looks at the 7 days *prior* to today.

### How to run the v3 preprocessing script
```bash
python src/preprocess_future_sequences_v3.py
```
Check `data/processed/` for the new `synthetic_future_sequences_v3.npz` file containing the enriched dataset.

## Phase 1.9: Train LSTM v3 with Seasonal Lag Features

### How LSTM v3 differs from LSTM v2

| Aspect | LSTM v2 (`train_lstm_future.py`) | LSTM v3 (`train_lstm_v3.py`) |
|---|---|---|
| Past features | 6 (units_sold, price, stock, promotion, day_of_week, month) | **8** (+`is_weekend`, +`rolling_7_day_average`) |
| Future features | 4 (price, promotion, day_of_week, month) | **7** (+`is_weekend`, +`same_day_last_week_sales`, +`rolling_7_day_average`) |
| Future encoder output | 16 dims | **32 dims** (wider to handle richer input) |
| Prediction head | 80 → 32 → 1 | **96 → 64 → 1** (deeper for more capacity) |
| Training epochs | 120 | **150** |
| Dropout in future encoder | ✗ | ✓ 0.1 |
| Dropout in prediction head | ✗ | ✓ 0.1 |

The core two-stream architecture (LSTM for the past + Linear encoder for the future, fused per forecast day) remains the same. V3 enriches both streams with stronger seasonal signals.

### Why `same_day_last_week_sales` helps

In Phase 1.5, the **Weekly Seasonal Baseline** (copy-last-week's-sales) was hard to beat because the LSTM had to reconstruct that weekly cycle from raw historical sequences. `same_day_last_week_sales` hands the model the exact answer the baseline uses — the sales from the same weekday 7 days ago — as an explicit feature.

Instead of spending its capacity re-discovering the "Monday always sells ~X units" pattern, the LSTM can now focus on *adjusting* that weekly baseline up or down based on price changes, promotions, and other signals. The model combines the strength of a heuristic baseline with the adaptability of deep learning.

### Why `rolling_7_day_average` helps

Daily sales data is noisy — a single unusually high or low day can distort what the model learns. The 7-day rolling average smooths that noise by summarising recent activity as a stable trend estimate. It acts like a built-in noise filter:

- **In `X_past`**: tells the model "the recent weekly trend is X units/day" rather than forcing it to infer this from seven individually noisy data points.
- **In `X_future`**: gives the model a stable reference point for each forecast day, anchoring predictions to the recent sales rate even when promotions or weekends push individual days up or down.

Because we apply a `shift(1)` before computing the rolling window, it only looks at the 7 days *before* today — no future data leaks in.

### Why `is_weekend` helps

Retail sales are structurally different on weekends. Saturday and Sunday sales can be 20–40 % higher than weekday averages in many product categories. `day_of_week` (0–6) encodes this implicitly, but it forces the model to learn that values 5 and 6 both mean "high sales day." `is_weekend` makes this explicit with a simple binary flag (1 = Sat/Sun, 0 = Mon–Fri), reducing the learning burden and improving generalisation on small datasets.

It is included in **both** `X_past` (so the model sees how weekends in recent history behaved) and `X_future` (so it knows which of the 7 upcoming forecast days fall on weekends).

### How to run the training script

1. Ensure Phase 1.8 preprocessing has been run first:
   ```bash
   python src/preprocess_future_sequences_v3.py
   ```

2. Train the LSTM v3 model from the `ml-service` directory:
   ```bash
   python src/train_lstm_v3.py
   ```

3. The script will:
   - Print data shapes and model architecture
   - Display training loss every 10 epochs (150 total)
   - Report MAE, RMSE, and MAPE on the held-out test set
   - Show one sample of actual vs. predicted next-7-days sales

4. Check the output files:
   - `models/lstm_v3_seasonal_model.pth` — saved model weights
   - `reports/lstm_v3_prediction_vs_actual.png` — prediction chart (single sample + test average)
   - `reports/lstm_v3_metrics.txt` — all evaluation metrics

## Phase 1.10: Regularized LSTM v3 with Validation Split and Early Stopping

### What overfitting is

Overfitting happens when a model learns the training data **too well** — including its noise, random fluctuations, and dataset-specific quirks — and then fails to generalise to unseen data. The tell-tale sign is a large gap between training loss and test loss:

- **Training loss**: very low (the model memorised the training examples).
- **Test / validation loss**: much higher (the memorised patterns don't exist in new data).

In Phase 1.9, LSTM v3's training MSE reached **0.002** while the test MSE was **0.046** — a 23× gap. That is a clear overfitting problem.

### Why validation split is needed

The test set must stay completely hidden until the very final evaluation. If we used the test set to decide when to stop training, we would be indirectly optimising for it and the test results would be misleadingly good.

Instead, we carve out a **validation set** from the training data:

| Subset | Source | Size | Purpose |
|---|---|---|---|
| Train | First 80% of original train | ~205 samples | Gradient updates, weight learning |
| Validation | Last 20% of original train | ~52 samples | Monitor generalisation during training |
| Test | Separate held-out set | 65 samples | Final unbiased evaluation (touched only once) |

**Chronological split is mandatory for time-series.** If we randomly sampled the validation set, future timestamps would leak into the training set, giving the model an unfair advantage. We always use the *earlier* portion for training and the *later* portion for validation.

### What early stopping does

Early stopping is a regularisation technique that automatically halts training when the model starts overfitting:

1. After every epoch, compute **validation loss** (no gradient updates, dropout disabled).
2. If validation loss is the new minimum → save a checkpoint of the model weights, reset patience counter to 0.
3. If validation loss does **not** improve → increment patience counter.
4. When patience counter reaches the threshold (20 epochs) → stop training and **restore the best checkpoint**.

The restored checkpoint represents the epoch where the model generalised best to unseen data, even though the training loss could have been driven lower with more epochs.

### Why `weight_decay` helps

`weight_decay=1e-4` adds an **L2 regularisation** penalty inside the Adam optimiser. At every weight update:

```
new_weight = old_weight × (1 − lr × weight_decay) − lr × gradient
```

The `(1 − lr × weight_decay)` factor gently shrinks every weight toward zero by a tiny fraction each step. This prevents the model from growing large, specialised weights that encode specific training quirks. Smaller weights → smoother decision surfaces → better generalisation.

`1e-4` is a mild setting. It barely affects convergence speed but provides a consistent nudge toward simpler solutions throughout training.

### Why lower validation loss matters more than lower training loss

Training loss measures **memory**. Validation loss measures **understanding**.

A model that memorises every training example will always score a lower training loss, but it has not learned the underlying demand patterns — it has learned the specific noise in that dataset. Because real-world demand forecasting operates on future, unseen data, only validation (and test) loss predicts real deployment performance.

The correct goal is not to minimise training loss. It is to minimise the **gap** between training loss and validation loss while keeping validation loss as low as possible.

### How to run the script

1. Ensure Phase 1.8 preprocessing has been run first:
   ```bash
   python src/preprocess_future_sequences_v3.py
   ```

2. Run the regularized training script from the `ml-service` directory:
   ```bash
   python src/train_lstm_v3_regularized.py
   ```

3. The script will:
   - Split training data chronologically (80% train / 20% validation)
   - Train with dropout=0.3, weight_decay=1e-4
   - Apply early stopping with patience=20
   - Restore the best checkpoint and evaluate on the test set

4. Check the output files:
   - `models/lstm_v3_regularized_best_model.pth` — best checkpoint (by validation loss)
   - `reports/lstm_v3_regularized_loss_curve.png` — training vs. validation loss chart
   - `reports/lstm_v3_regularized_prediction_vs_actual.png` — prediction chart
   - `reports/lstm_v3_regularized_metrics.txt` — all metrics + regularisation config

## Phase 1.11: LSTM v4 Residual Forecasting Model

### What residual forecasting is

Residual forecasting is a hybrid modeling approach that combines a simple, reliable baseline forecast (often a linear or seasonal heuristic) with a complex machine learning model (like an LSTM). Instead of predicting the final target value directly, the neural network is trained to predict the **residual error** (or correction) of the baseline:

$$\text{Residual} = \text{Actual Sales} - \text{Baseline Forecast}$$

During inference, we compute the final prediction by adding the network's predicted residual back to the baseline:

$$\text{Final Prediction} = \text{Baseline Forecast} + \text{Predicted Residual}$$

### Why we use weekly baseline as a starting point

In retail and inventory demand, weekly seasonal patterns are extremely dominant. For instance, weekend sales peaks and weekday troughs repeat week-over-week with high consistency. A simple heuristic like "predicting same-day last-week's sales" (the Weekly Seasonal Baseline) acts as an incredibly strong baseline that captures this pattern perfectly.

By using this baseline as the starting point, we give the model a massive head start. Rather than forcing the neural network to spend its entire optimization capacity learning basic calendar dynamics (such as "Saturday sales are higher"), we inject this structural knowledge directly.

### Why the neural network predicts correction instead of full demand

When a neural network predicts absolute sales directly, it has to learn:
1. The global scale of sales.
2. Strong periodic seasonality (weekly/monthly/yearly).
3. The impact of localized, dynamic events (promotions, price fluctuations, out-of-stock events).

This is a complex function mapping. When predicting the **residual correction**, the baseline already handles (1) and (2). The network only has to learn (3) — how promotions, price drops, or rolling trends cause demand to *deviate* from its typical weekly pattern.

By predicting the residual correction:
- The target distribution is zero-centered and has lower variance.
- The model is less prone to catastrophic scale errors (e.g., predicting negative sales or massive spikes).
- Optimization becomes significantly easier since the model is approximating a simpler function.

### Why this is useful for business forecasting

In a business context, demand planning and supply chain systems must be both **accurate** and **robust**:
- **Robustness**: If a machine learning model experiences a sudden distribution shift or sequence failure, predicting absolute values can lead to severe over- or under-stocking. A residual model guarantees that, at worst, the model reverts back to a highly reliable weekly seasonal baseline.
- **Explainability**: Planners can inspect predictions easily. They can see: *"The baseline forecast is 100 units based on last week, and the ML model added a residual of +15 units because we scheduled a promotion on this day."* This transparent decomposition builds trust and facilitates manual inventory overrides.

### How to run the scripts

1. Preprocess residual sequences:
   ```bash
   python src/preprocess_residual_sequences.py
   ```

2. Train the residual LSTM model:
   ```bash
   python src/train_lstm_residual.py
   ```

3. Output files:
   - `models/lstm_v4_residual_model.pth` — trained weights for predicting residuals.
   - `reports/lstm_v4_residual_prediction_vs_actual.png` — comparison chart.
   - `reports/lstm_v4_residual_metrics.txt` — metrics comparison (LSTM Residual vs Heuristic Baseline).

## Phase 1 Summary & Project Restructuring

### Synthetic Phase Recap

Phase 1 focused on learning the fundamentals of time-series demand forecasting using a generated, synthetic dataset (`sample_sales.csv`). We progressively built more sophisticated LSTM models, learning key ML concepts along the way.

**Best Synthetic Model**: 
The **LSTM v3 Regularized** model performed best overall, achieving an MAPE of 17.26%. It successfully balanced learning seasonal patterns (weekends, rolling averages) and future known features (price, promotions) without overfitting, thanks to the validation split and early stopping.

**Best Baseline**:
The **Weekly Seasonal Baseline** remained extremely competitive, highlighting the importance of strong heuristics in retail forecasting.

### Why we are moving to real data

Synthetic data is perfect for learning the mechanics of sequence modeling because it is clean and mathematically predictable. However, real-world retail data is incredibly noisy. It contains missing days, extreme outliers, holiday effects, store closures, and complex interacting trends that a synthetic generator cannot fully capture. 

Phase 2 will tackle the real-world **Rossmann Store Sales** dataset to apply these techniques to actual, noisy, multi-store business data.

### New Project Structure

To prepare for Phase 2, the `ml-service` project has been restructured to cleanly separate synthetic learning scripts from real-world modeling scripts.

```text
ml-service/
├── data/
│   ├── synthetic/      # All Phase 1 sample_sales data
│   │   ├── raw/
│   │   └── processed/
│   └── real/           # Upcoming Phase 2 Rossmann data
│       ├── raw/
│       └── processed/
├── models/
│   ├── synthetic/      # Phase 1 trained weights (.pth) and scalers (.pkl)
│   └── real/
├── reports/
│   ├── synthetic/      # Phase 1 charts and metrics
│   └── real/
└── src/
    ├── synthetic/      # All Phase 1 training and preprocessing scripts
    ├── real/           # Upcoming Phase 2 scripts
    └── shared/         # Reusable helper functions
```

## Phase 2.1: Prepare the Real Rossmann Store Sales Dataset

### Why we moved from synthetic to real data

Phase 1 used a hand-crafted, perfectly clean, single-product synthetic dataset. While it was ideal for learning LSTM mechanics, it had important limitations:

- **No missing data, outliers, or noise** — real data is messy.
- **No holidays, store closures, or irregular gaps** — real retail has all of these.
- **Single-product / single-store** — real retail involves thousands of SKUs and stores.
- **Predictable patterns** — our synthetic data generator explicitly coded weekly/monthly cycles, so the model already "knew" the generating process.

The **Rossmann Store Sales** dataset is a well-known Kaggle competition dataset containing 2.5 years of daily sales data across 1,115 drugstores in Germany. It includes promotions, school holidays, state holidays, competition distance, and much more — providing a realistic testbed for demand forecasting.

### Why `Customers` is not used as an input feature

The Rossmann dataset includes a `Customers` column (the number of customers who visited the store on each day). This column is **strongly correlated with Sales** — more customers almost always means more revenue.

However, using it as a feature would be a critical mistake called **target leakage**.

### What target leakage means

**Target leakage** occurs when your training features contain information that would not be available at the time of prediction. In other words, the model "cheats" by using future or contemporaneous information that it would never have in a real deployment.

Example with `Customers`:
- The model needs to predict *next Tuesday's sales*.
- It cannot know how many customers will walk in next Tuesday — that number only becomes available *after* Tuesday is over.
- If we train with `Customers` as an input, the model learns: "just multiply customer count by average spend per customer" — an almost perfect predictor.
- At inference time, `Customers` doesn't exist, and the model's predictions collapse.

**Rule of thumb:** Before including any feature, ask: *"Would I know this value BEFORE I need to make the prediction?"* If the answer is no, exclude it.

### Why closed-store rows are removed

When a store is closed (`Open == 0`), sales are always 0. Including these rows would:

1. **Bias the model** — the average sales drops, and the model learns a trivial "closed = 0" pattern that wastes its capacity.
2. **Confuse sequence windows** — an LSTM sliding over a history that alternates between real sales and zeros will learn noise, not signal.
3. **Not matter in production** — store opening/closing schedules are known in advance. The application layer would simply output 0 for closed days and only invoke the ML model for open days.

For the first model, we remove closed-day rows entirely so the model can focus on learning the demand patterns of open days.

### Why we start with one store first

The full Rossmann dataset has 1,115 stores. Jumping straight to multi-store modeling introduces massive complexity:

- Each store has different demand patterns, local holidays, and competition effects.
- Training time and memory usage increase dramatically.
- Debugging becomes almost impossible when predictions are wrong — is it the model architecture? The data for that store? A bug in the pipeline?

By starting with **Store 1 only** (781 open-day rows), we can:
1. Validate the entire pipeline end-to-end.
2. Deeply inspect charts and predictions.
3. Build intuition about real-data quality before scaling.

### How to run the script

1. Place the raw Rossmann files in the expected location:
   ```
   data/real/raw/train.csv
   data/real/raw/store.csv
   ```

2. Run the preparation script from the `ml-service` directory:
   ```bash
   python src/real/prepare_rossmann_data.py
   ```

3. The script will:
   - Load and merge train.csv with store.csv
   - Filter to Store 1 and remove closed-day rows
   - Handle missing values and one-hot encode categoricals
   - Save the processed CSV and generate EDA charts

4. Output files:
   - `data/real/processed/rossmann_store_1_processed.csv` — clean, ready-to-use dataset
   - `reports/real/rossmann_store_1_sales_over_time.png` — daily sales timeline
   - `reports/real/rossmann_store_1_sales_by_day_of_week.png` — weekly seasonality
   - `reports/real/rossmann_store_1_promo_effect.png` — promotion impact
   - `reports/real/rossmann_store_1_monthly_sales.png` — monthly patterns

### How to verify raw dataset files

Before running `prepare_rossmann_data.py`, you can quickly check that the required files are present and inspect their schemas:

```bash
python src/real/check_real_dataset_files.py
```

This prints file sizes, column names, and the first 5 rows of both `train.csv` and `store.csv`.

## Phase 2.2: Create Real LSTM-Ready Sequences

### What real-data sequences mean and how they differ from synthetic
Unlike our synthetic dataset, which was perfectly continuous, real data can have gaps, missing days, and irregular intervals (especially after removing closed-store days). Creating sequences on real data requires careful handling of missing values. Furthermore, real data has complex holiday and promotional schedules that we must correctly separate into past and future inputs without data leakage.

### What lag_7_sales means
`lag_7_sales` is a feature that represents the sales from exactly 7 days prior. Since retail demand strongly depends on the day of the week, the best predictor of today's sales is often the sales from the same day last week. This explicit historical feature helps the LSTM model anchor its prediction.

### What rolling_7_sales means
`rolling_7_sales` is the rolling average of sales over the previous 7 days. It smooths out day-to-day noise and provides the LSTM with a stable recent trend. We use `shift(1)` before applying the rolling window to ensure that future values (like today's sales) do not leak into the average.

### Why we calculate baseline metrics before training LSTM
Before training a complex deep learning model like an LSTM, we must establish baseline metrics using simple heuristics (like the Last Value or Weekly Seasonal baseline). These baselines act as a benchmark. If our complex LSTM cannot beat a simple rule like "predict last week's sales," then the model is not learning effectively or the architecture is flawed. Baselines justify the need for ML.

### How to run the script
1. Ensure the processed real dataset is available:
   ```bash
   data/real/processed/rossmann_store_1_processed.csv
   ```
2. Run the sequence creation script from the `ml-service` directory:
   ```bash
   python src/real/preprocess_rossmann_sequences.py
   ```
3. The script will output dataset statistics, sequences, and scaler files.
4. Baseline metrics will be saved to:
   ```bash
   reports/real/rossmann_store_1_baseline_metrics.txt
   ```

## Phase 2.3: Train a Real Future-Aware LSTM

### How real-data LSTM differs from synthetic LSTM
Training an LSTM on real data introduces complexities that synthetic data generators mask. Real data sequences have noisy patterns that change across time. The real-data LSTM uses actual past observations combined with scheduled future events (like holidays and promotions) to make its predictions. Unlike synthetic data which is uniform, the model must learn robust features to handle variability in actual store operations.

### Why validation split is used
In Phase 1, we learned that time-series data must be split chronologically. The same applies for our validation set. Instead of evaluating blindly on the test set, we separate the training data into an 80% training portion and a 20% validation portion. The validation set monitors generalisation during training, making sure the model actually understands the underlying sales patterns and doesn't just memorise the noise of the training data.

### Why early stopping is used
Neural networks will continue to memorise data (reduce training loss) long after they stop learning useful generalisations. Early stopping monitors the validation loss and halts training when it stops improving, preventing overfitting. We then restore the weights from the best epoch so our final model is optimally tuned.

### Why the model must beat the baseline
Deep learning models are complex and resource-intensive. If a neural network cannot outperform a naive heuristic like the Last Value Baseline (MAPE = 20.96%) or the Weekly Seasonal Baseline (MAPE = 25.79%), it is not worth using. Our real LSTM model explicitly aims to combine baseline strength with deep learning adaptability to surpass these marks.

### How to run the script
```bash
python src/real/train_rossmann_lstm.py
```
Check `reports/real/` for prediction charts, loss curves, and the saved model metrics.

## Phase 2.4: Detailed Evaluation of Real LSTM

### Why overall metrics are not enough
Overall metrics like a single MAPE of 7.76% provide a good high-level summary, but they can hide critical weaknesses. For example, a model might predict the first few days very accurately but fail completely on days further into the future. It could also have a low average error but occasionally make catastrophic predictions. We need detailed evaluation to uncover these hidden behaviors.

### Why forecast-day-level error matters
Time-series forecasting typically becomes harder the further out you predict. By breaking down MAE, RMSE, and MAPE by forecast day (Day 1 through Day 7), we can observe how quickly the model's accuracy degrades. If error spikes drastically by Day 7, the model may only be reliable for short-term planning.

### Why worst predictions are useful
In business demand forecasting, extreme errors (e.g., predicting 5000 sales on a day with 0 sales) cause the most disruption, such as massive overstock or severe stockouts. Extracting the worst 10 predictions allows human analysts to investigate why the model failed. Was there an unrecorded local holiday? A data glitch? Understanding the worst cases is key to improving robustness.

### What it means that the LSTM beats both baselines
A machine learning model is only valuable if it outperforms simple heuristics. Since the LSTM's MAPE (7.76%) is substantially lower than both the Last Value Baseline (20.96%) and the Weekly Seasonal Baseline (25.79%), we have proven that the neural network has successfully learned complex, non-trivial patterns from the real-world data (such as the impact of promotions and complex seasonality) that basic rules cannot capture.

## Phase 2.5: Worst-Prediction Diagnosis with Dates and Business Context

### Why worst prediction diagnosis matters
An aggregate MAPE of 7.76% hides individual failure modes. In business forecasting, a single day where we predict 4,600 units but actually sell 6,700 units triggers a stock-out that costs real revenue. Diagnosis pins down exactly which calendar dates caused the worst errors, enabling targeted model improvement.

### Why overlapping windows can repeat the same difficult date
The sliding-window sequence method creates many overlapping sequences from the same timeline. For example, if 2015-04-04 was an exceptional high-sales day (Easter Saturday with school holiday), it appears as the target of sequences with sample_index 57 (day 1), 56 (day 2), 55 (day 3), etc. This is why the worst predictions CSV shows the same date `2015-04-04` with `actual_sales ≈ 6709` repeated across multiple rows — they all point at that one calendar date from different sequence windows.

### How business context helps improve the model
After running diagnosis on the real Store 1 data we discovered:
- **April had the highest average error** (MAE ≈ 597) — driven by Easter public holidays in 2015. The model had no direct public holiday flag to warn it.
- **Promotion days have ~30% higher errors** than non-promotion days (402 vs 311), suggesting the model underestimates the full magnitude of some promotions.
- **Saturdays (Day 6) are hardest to predict** (MAE ≈ 437) vs Mondays (MAE ≈ 240), because weekend sales are more volatile.

These findings directly guide the next iteration: add a `StateHoliday` or `PublicHoliday` binary flag to both past and future feature columns.

### How to run the script
```bash
python src/real/diagnose_rossmann_errors.py
```
Output files:
- `reports/real/rossmann_store_1_worst_predictions_with_context.csv` — worst 20 errors with calendar context
- `reports/real/rossmann_store_1_error_context_summary.txt` — group error averages by promo, weekday, month
- `reports/real/rossmann_store_1_error_by_promo.png`
- `reports/real/rossmann_store_1_error_by_weekend.png`
- `reports/real/rossmann_store_1_error_by_day_of_week.png`
- `reports/real/rossmann_store_1_error_by_month.png`

## Phase 2.6: Enhanced Real-Data Preprocessing with Event/Context Features

### Why event/context features were added
Phase 2.5 diagnosis revealed that the model's worst errors cluster around specific calendar conditions — Easter week in April, promo + school holiday coincidences, and volatile Saturdays. The v1 model had no explicit flags for these conditions. Adding dedicated features lets the LSTM immediately identify high-risk days instead of guessing from raw noisy sales history alone.

### Why April was added as a dedicated feature (`is_april`)
Of all months, April has by far the highest average prediction error (MAE ≈ 597 vs. an overall average of ≈ 351). The root cause is Easter, which falls in April and drives an unusually large sales spike across multiple consecutive days. The `Month` column already encodes April as the value 4, but an explicit `is_april` binary flag gives the LSTM an unambiguous, high-salience signal that it is entering a structurally different demand regime.

### Why interaction features help
Retail demand does not respond to events independently. A promotion alone raises demand. A school holiday alone raises demand. But when both are active simultaneously (`promo_schoolholiday = 1`), the combined effect is larger than the sum of the parts — parents are at home, children are available, and discounts are active. Providing the product of two binary features directly removes the need for the LSTM to discover this joint effect through many training examples.

| Interaction | Business interpretation |
|---|---|
| `promo_schoolholiday` | Marketing campaign active during a school break |
| `weekend_schoolholiday` | Weekend AND school holiday — both parents and children at home |
| `promo_weekend` | Weekend promotion — peak footfall day with a discount |

### Why rolling features must not leak future values
If we calculate `rolling_7_sales` directly on the raw `Sales` column without `shift(1)`, then at row `i` the rolling window includes `Sales[i]` — the very value we are trying to predict. The model would learn to copy today's actual sales into its rolling feature, achieving near-zero training loss while learning nothing useful. `shift(1)` pushes every value one row forward, so the window at row `i` only sees rows `i-1` through `i-7`.

### Why future Sales is not allowed in `X_future`
`X_future` represents information we hand the model for the 7 days it is about to forecast. If we included `Sales` in that window, we would be handing the model the exact answers it is supposed to compute. In deployment, those values don't exist yet. This form of contamination is called **data leakage** and is one of the most dangerous mistakes in applied ML — it produces misleadingly good metrics that collapse the moment the model is used on new data.

### How to run the script
```bash
python src/real/preprocess_rossmann_sequences_v2.py
```
Output files:
- `data/real/processed/rossmann_store_1_sequences_v2.npz` — enhanced sequences (731 total: 584 train, 147 test)
- `models/real/rossmann_v2_past_feature_scaler.pkl`
- `models/real/rossmann_v2_future_feature_scaler.pkl`
- `models/real/rossmann_v2_target_scaler.pkl`
- `reports/real/rossmann_store_1_v2_feature_list.txt` — full feature manifest

**v1 vs v2 feature comparison:**

| | v1 | v2 |
|---|---|---|
| Past features | 9 | **20** |
| Future features | 8 | **18** |
| Rolling windows | 7-day avg | 7-day avg, **14-day avg, 7-day std** |
| Event flags | — | **is_month_start, is_month_end, is_april** |
| Interaction terms | — | **promo×school, weekend×school, promo×weekend** |
| Promo timing | — | **days_since_last_promo, days_until_next_promo** |
| Momentum | — | **sales_vs_lag7 (past only)** |

## Phase 2.7: Train LSTM v2 with Enhanced Event/Context Features

### What new features were added in v2
Based on error diagnosis from Phase 2.5, eleven features were added to address the specific failure conditions of the v1 model:

| Category | New features |
|---|---|
| Extended rolling | `rolling_14_sales`, `rolling_7_std_sales` |
| Calendar events | `is_month_start`, `is_month_end`, `is_april` |
| Interaction terms | `promo_schoolholiday`, `weekend_schoolholiday`, `promo_weekend` |
| Promo timing | `days_since_last_promo`, `days_until_next_promo` |
| Momentum | `sales_vs_lag7` (past only) |

### Why event/context features may help
Each new feature targets a specific failure pattern identified by diagnosis. `is_april` alerts the model to Easter-season volatility. `promo_schoolholiday` captures multiplicative demand amplification when promotions overlap with school breaks. `days_until_next_promo` models demand anticipation — shoppers may stock up in the days immediately before a promotion ends. These signals reduce the model's need to infer complex patterns from noisy raw sales alone.

### Why more features can also cause overfitting
Adding features expands the model's capacity to fit the training data. With 20 past features and 18 future features (versus 9 and 8 in v1), the model has many more parameters to tune. Without regularisation, these extra degrees of freedom would quickly memorise training-set noise. The v2 model manages this risk through: dropout (0.3 in LSTM, 0.2 in encoders), L2 weight decay (1e-4), and early stopping.

### Why early stopping is used
The v2 model stopped at epoch 103 with best weights at epoch 78. Without early stopping, the extra feature capacity would have driven training loss lower while validation loss rose — the classic overfitting signature. Early stopping captured the generalisation peak automatically.

### Results — v2 vs v1 vs Baselines

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Last Value Baseline | 935.54 | 1176.05 | 20.96% |
| Weekly Seasonal Baseline | 1135.91 | 1389.58 | 25.79% |
| **LSTM v1** | 351.74 | 479.05 | 7.76% |
| **LSTM v2** | **339.21** | **455.63** | **7.51%** |

v2 improves over v1 by **0.25 percentage points** in MAPE, confirming that the event/context features provide a genuine and measurable signal improvement.

### How to run the script
```bash
python src/real/train_rossmann_lstm_v2.py
```
Output files:
- `models/real/rossmann_store_1_lstm_v2_model.pth`
- `reports/real/rossmann_store_1_lstm_v2_prediction_vs_actual.png`
- `reports/real/rossmann_store_1_lstm_v2_loss_curve.png`
- `reports/real/rossmann_store_1_lstm_v2_metrics.txt`

## Phase 2.8: Detailed Evaluation and Final Champion Report (LSTM v2)

### Final Store 1 Champion: LSTM v2

| Model | MAE | RMSE | MAPE | vs Baselines |
|---|---|---|---|---|
| Last Value Baseline | 935.54 | 1176.05 | 20.96% | — |
| Weekly Seasonal Baseline | 1135.91 | 1389.58 | 25.79% | — |
| LSTM v1 | 351.74 | 479.05 | 7.76% | -13.20 pp vs LV |
| **LSTM v2 (Champion)** | **332.43** | **439.03** | **7.32%** | **-13.64 pp vs LV** |

LSTM v2 beats the Last Value Baseline by **13.64 percentage points** in MAPE and improves over v1 by **0.44 pp**.

### What improved from v1 to v2

| Category | Change |
|---|---|
| Past features | 9 → 20 (added rolling 14-day, std, interaction terms, promo timing, momentum) |
| Future features | 8 → 18 (same additions minus sales-dependent momentum) |
| Future encoder | 1-layer Linear → 2-layer MLP (handles non-linear interactions better) |
| April signal | Not present in v1 → explicit `is_april` flag reduces Easter errors |
| Forecast-day stability | v1 MAPE varied slightly across days; v2 is flat (7.27%–7.37%) |

### Forecast-day stability in v2
Unlike many LSTM models where error increases with the forecast horizon (day 7 is harder than day 1), v2 shows remarkably flat MAPE across all 7 days (7.27% to 7.37%). This means the model is equally reliable for same-week planning decisions whether the target day is tomorrow or in 7 days.

### Output files generated by Phase 2.8
- `reports/real/rossmann_store_1_lstm_v2_detailed_predictions.csv` — all 1,029 test predictions with errors
- `reports/real/rossmann_store_1_lstm_v2_worst_predictions.csv` — top 20 worst errors for analyst review
- `reports/real/rossmann_store_1_lstm_v2_error_by_forecast_day.png` — per-day MAE bar chart
- `reports/real/rossmann_store_1_lstm_v2_actual_vs_predicted_all_test.png` — flattened timeline comparison
- `reports/real/rossmann_store_1_lstm_v2_model_comparison_mape.png` — headline comparison of all 4 models
- `reports/real/rossmann_store_1_champion_model_summary.txt` — official champion model record

### How to run the evaluation
```bash
python src/real/evaluate_rossmann_lstm_v2_detailed.py
```
No retraining is needed. The script loads the saved weights and scaler and generates all reports from scratch.

## Phase 2.9: Champion Model Inference Pipeline

### Difference between training and inference

| Aspect | Training | Inference |
|---|---|---|
| Historical sales | Used as model targets (labels) | Used only as context (X_past) |
| Future sales | Known (in dataset) | Unknown -- model predicts these |
| Scalers | Fit on training data | Loaded from saved .pkl files |
| Output | Learns weights | Generates a 7-day sales forecast |

During training the model saw millions of examples where future Sales were available. During inference, those values do not exist yet -- the model must predict them from the historical context and the known business plan.

### Why the model needs the latest 30 days (X_past)
The LSTM encoder compresses the 30-day window into a 64-dimensional context vector. This vector carries the model's "memory" of recent demand: trend direction, recent promo effects, volatility. Without this context, every forecast would start from the same prior and produce identical outputs for every store and every date.

### Why future known features are required (X_future)
The second model pathway reads 7 days of known business context: promotions, holidays, weekday patterns. These are available before the forecast period because the business plans them in advance. Without them, the model would have to guess all event context, severely underperforming on high-impact days like promo weekends or school holidays.

### Why inference must match training assumptions (The "Closed Days" Problem)
A machine learning model only knows the universe it was trained on. During training, we removed all rows where `Open == 0` (mostly Sundays). The model *never saw a closed day*. 

If we naively ask the model to predict a Sunday, it will output a normal open-day sales volume (e.g., 4,362 units), which is physically impossible for a closed store. To fix this, the inference pipeline:
1. Detects if the store historically closes on Sundays.
2. Supports a `FORECAST_OPEN_DAYS_ONLY` mode to skip known closed days and forecast the next 7 *real business days*.
3. If forecasting calendar days, overrides the model's prediction to `0` for closed days.

### Scaler alignment and DayOfWeek encoding
Feature values passed at inference time MUST use the exact same encoding and numerical range as those seen during training. Any feature that falls outside the trained scaler's min/max range produces scaled values outside [0,1], which the model has never seen and which produce garbage predictions.
- **DayOfWeek**: Rossmann encoding uses `1=Monday` to `7=Sunday`. The `Date.weekday()` must be adjusted (`+ 1`) so that DayOfWeek features map correctly to what the scaler expects.
- **days_since/until_next_promo**: training max = 10. Values are capped at 10 to stay in distribution.
- **Safety clip**: all scaled values are hard-clipped to [0, 1] before entering the model to prevent silent out-of-distribution errors.

### Sample output (Aug 2015 forecast, open days only)
```
Date           Weekday    Promo    Predicted Sales
--------------------------------------------------
2015-08-01     Sat        No              4,743 <<< PEAK
2015-08-03     Mon        No              3,439 <<< LOW
2015-08-04     Tue        No              3,450
2015-08-05     Wed        No              3,450
2015-08-06     Thu        No              3,457
2015-08-07     Fri        No              3,488
2015-08-08     Sat        No              4,047
--------------------------------------------------
Total predicted sales:                    26,076
Average predicted sales:                   3,725
```

### How to run the prediction script
```bash
python src/real/predict_rossmann_store_1.py
```
Output files:
- `reports/real/rossmann_store_1_next_7_day_forecast.csv` -- tabular forecast
- `reports/real/rossmann_store_1_next_7_day_forecast.png` -- bar chart forecast

To customise the 7-day plan, edit the `Promo` and `SchoolHoliday` defaults in the `future_rows` loop inside the script, or toggle `FORECAST_OPEN_DAYS_ONLY` at the top of the file.
