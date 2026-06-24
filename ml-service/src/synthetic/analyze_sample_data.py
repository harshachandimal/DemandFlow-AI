import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Ensure the reports directory exists
    os.makedirs('reports', exist_ok=True)

    # 2. Load the dataset
    data_path = 'data/sample_sales.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run the generation script first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # Parse the date column as datetime
    # Purpose: Time-series models require a properly formatted temporal index to understand sequence.
    df['date'] = pd.to_datetime(df['date'])

    # Print dataset shape
    # Purpose: Understand the total number of samples and features available for training.
    print("\n--- Dataset Shape ---")
    print(df.shape)

    # Print first 5 rows
    # Purpose: Sanity check to ensure data loaded correctly and features look as expected.
    print("\n--- First 5 Rows ---")
    print(df.head())

    # Print missing values per column
    # Purpose: ML models generally cannot handle missing data (NaNs). We must identify them before training.
    print("\n--- Missing Values Per Column ---")
    print(df.isnull().sum())

    # units_sold average, min, max
    # Purpose: Understand the scale and range of our target variable to help with scaling/normalization later.
    print("\n--- Units Sold Statistics ---")
    print(f"Average: {df['units_sold'].mean():.2f}")
    print(f"Min: {df['units_sold'].min()}")
    print(f"Max: {df['units_sold'].max()}")

    # average sales by day_of_week
    # Purpose: Identify weekly seasonality. If certain days consistently sell more, the model needs to learn this pattern.
    print("\n--- Average Sales by Day of Week ---")
    day_of_week_avg = df.groupby('day_of_week')['units_sold'].mean()
    print(day_of_week_avg)

    # average sales when promotion = 1 vs promotion = 0
    # Purpose: Quantify the impact of marketing events. This tells us if 'promotion' is a strong predictive feature.
    print("\n--- Promotion Effect on Sales ---")
    promo_avg = df.groupby('promotion')['units_sold'].mean()
    print(f"Promotion = 0 (No): {promo_avg.get(0, 0):.2f}")
    print(f"Promotion = 1 (Yes): {promo_avg.get(1, 0):.2f}")

    # average sales by month
    # Purpose: Identify yearly seasonality or longer-term trends across different months.
    print("\n--- Average Sales by Month ---")
    month_avg = df.groupby('month')['units_sold'].mean()
    print(month_avg)

    # 4. Create and save charts
    print("\nGenerating charts in reports/ directory...")

    # line chart of date vs units_sold
    # Purpose: Visualize the overall trend, seasonality, and variance across the entire timeline.
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['units_sold'], marker='o', linestyle='-', markersize=2)
    plt.title('Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/sales_over_time.png')
    plt.close()

    # bar chart showing average units_sold per day_of_week
    # Purpose: Visually confirm weekly seasonal patterns.
    plt.figure(figsize=(8, 5))
    day_of_week_avg.plot(kind='bar', color='skyblue')
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Average Units Sold')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('reports/sales_by_day_of_week.png')
    plt.close()

    # bar chart comparing promotion vs non-promotion average sales
    # Purpose: Visually highlight the feature importance of promotions.
    plt.figure(figsize=(6, 5))
    promo_avg.plot(kind='bar', color=['lightcoral', 'lightgreen'])
    plt.title('Effect of Promotion on Sales')
    plt.xlabel('Promotion (0=No, 1=Yes)')
    plt.ylabel('Average Units Sold')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('reports/promotion_effect.png')
    plt.close()

    # bar chart showing average units_sold per month
    # Purpose: Visually confirm monthly or broader seasonal shifts.
    plt.figure(figsize=(10, 5))
    month_avg.plot(kind='bar', color='orange')
    plt.title('Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Units Sold')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('reports/monthly_sales_pattern.png')
    plt.close()

    print("Exploratory Data Analysis complete! Charts saved to reports/.")

if __name__ == "__main__":
    main()
