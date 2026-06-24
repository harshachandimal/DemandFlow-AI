import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_sales_data(num_days=365):
    """
    Generates synthetic daily sales data for a specific product.
    
    This function simulates realistic business scenarios where sales are influenced
    by factors like day of the week, seasonality, promotions, and a general trend over time.
    """
    # 1. Initialize lists to hold our column data
    dates = []
    products = []
    units_sold_list = []
    prices = []
    stocks = []
    promotions = []
    days_of_week = []
    months = []
    
    # 2. Define starting parameters
    start_date = datetime(2023, 1, 1) # Start from a specific date
    product_name = "Rice 5kg"
    base_sales = 50       # Base demand: average units sold on a normal day
    base_price = 15.99    # Regular price
    base_stock = 1000     # Starting inventory
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        
        # --- Time Features ---
        # Machine learning models like LSTMs need explicit features to understand time patterns.
        # Extracting day_of_week (0=Monday, 6=Sunday) and month (1-12) helps the model 
        # learn weekly and monthly seasonality.
        day_of_week = current_date.weekday() 
        month = current_date.month
        
        # --- Sales Boost Factors ---
        # 1. Weekend Boost: Sales generally go up on weekends (Saturday=5, Sunday=6)
        weekend_boost = 0
        if day_of_week >= 5:
            weekend_boost = np.random.randint(20, 40) # Add 20-40 extra units on weekends
            
        # 2. Seasonal Boost: Let's assume higher demand in certain months (e.g., winter or holiday season)
        seasonal_boost = 0
        if month in [11, 12, 1]: # Higher demand in Nov, Dec, Jan
            seasonal_boost = np.random.randint(10, 30)
            
        # 3. Promotion Boost: Simulate marketing campaigns
        # There's a 10% chance of a promotion on any given day
        is_promotion = np.random.choice([True, False], p=[0.1, 0.9])
        promotion_boost = 0
        if is_promotion:
            promotion_boost = np.random.randint(30, 60) # Promotions significantly increase sales
            
        # 4. Upward Trend: Simulate business growth over the year
        # Small incremental increase as the days go by
        trend = int((day / num_days) * 20) # Up to 20 extra units by the end of the year
        
        # 5. Random Noise: Real-world data is never perfectly predictable. 
        # Add some random fluctuations (normal distribution)
        noise = int(np.random.normal(0, 5)) 
        
        # --- Calculate Final Target Variable (units_sold) ---
        # Combine all the factors to get the final sales figure
        units_sold = base_sales + weekend_boost + seasonal_boost + promotion_boost + trend + noise
        
        # Ensure units_sold is never negative (can't sell negative items)
        units_sold = max(0, units_sold)
        
        # --- Update Stock and Price ---
        # Basic inventory simulation: decrease stock by units sold
        # In a real scenario, there would be restocks
        current_stock = base_stock - units_sold
        if current_stock < 0:
            current_stock = 0 # Out of stock
            units_sold = base_stock # Can only sell what we have
            base_stock = 0
        else:
            base_stock = current_stock
            
        # Daily price: Slightly lower if on promotion
        price = round(base_price * 0.8, 2) if is_promotion else base_price
        
        # Append to our lists
        dates.append(current_date.strftime('%Y-%m-%d'))
        products.append(product_name)
        units_sold_list.append(units_sold)
        prices.append(price)
        stocks.append(current_stock)
        promotions.append(int(is_promotion)) # 1 for True, 0 for False (ML models prefer numbers)
        days_of_week.append(day_of_week)
        months.append(month)
        
        # Restock logic just so we don't stay at 0
        if base_stock < 100:
            base_stock += 1000

    # 3. Create a pandas DataFrame
    df = pd.DataFrame({
        'date': dates,
        'product': products,
        'units_sold': units_sold_list,
        'price': prices,
        'stock': stocks,
        'promotion': promotions,
        'day_of_week': days_of_week,
        'month': months
    })
    
    return df

if __name__ == "__main__":
    print("Starting data generation for Phase 1.1...")
    
    # Generate the data
    df_sales = generate_sales_data(365)
    
    # Determine the absolute path to the data directory
    # This ensures the script works regardless of where it's executed from
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / 'data'
    
    # Create the data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the output file path
    output_path = data_dir / 'sample_sales.csv'
    
    # Save to CSV
    df_sales.to_csv(output_path, index=False)
    
    # Print the required statistics
    print(f"\n--- Data Generation Complete ---")
    print(f"Dataset saved to: {output_path.resolve()}")
    print(f"\nTotal number of rows: {len(df_sales)}")
    print(f"\nFirst 5 rows:\n{df_sales.head()}")
    
    print(f"\n--- Statistics for 'units_sold' ---")
    print(f"Average units_sold: {df_sales['units_sold'].mean():.2f}")
    print(f"Max units_sold: {df_sales['units_sold'].max()}")
    print(f"Min units_sold: {df_sales['units_sold'].min()}")
