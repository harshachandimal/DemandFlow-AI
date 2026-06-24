from typing import List, Optional

def generate_business_insights(
    predictions: List[int],
    current_stock: Optional[int],
    unit_price: Optional[float],
    reorder_lead_time_days: int,
    safety_stock_percentage: float
) -> dict:
    # 1. Calculate total predicted sales
    total_predicted_sales = sum(predictions)
    
    # 2. Calculate average predicted sales
    if len(predictions) > 0:
        average_predicted_sales = round(total_predicted_sales / len(predictions))
    else:
        average_predicted_sales = 0
        
    # 3. Calculate expected revenue
    if unit_price is not None:
        expected_revenue = total_predicted_sales * unit_price
    else:
        expected_revenue = None
        
    # 4. If current_stock is not provided
    if current_stock is None:
        return {
            "total_predicted_sales": total_predicted_sales,
            "average_predicted_sales": average_predicted_sales,
            "expected_revenue": expected_revenue,
            "current_stock": None,
            "projected_stock_after_7_days": None,
            "stockout_risk": None,
            "recommended_reorder_quantity": None,
            "reorder_needed": None,
            "recommendation": "Forecast generated successfully. Add current_stock to receive stock-out and reorder recommendations."
        }
        
    # 5. Calculate projected stock
    projected_stock_after_7_days = current_stock - total_predicted_sales
    
    # 6. Calculate safety stock
    safety_stock = total_predicted_sales * safety_stock_percentage
    
    # 7. Calculate reorder point
    reorder_point = average_predicted_sales * reorder_lead_time_days + safety_stock
    
    # 8-10. Determine stockout risk and reorder needed
    if projected_stock_after_7_days < 0:
        stockout_risk = "high"
        reorder_needed = True
    elif current_stock <= reorder_point:
        stockout_risk = "medium"
        reorder_needed = True
    else:
        stockout_risk = "low"
        reorder_needed = False
        
    # 11. Calculate recommended reorder quantity
    if reorder_needed:
        recommended_reorder_quantity = max(0, round(total_predicted_sales + safety_stock - current_stock))
    else:
        recommended_reorder_quantity = 0
        
    # 12. Recommendation text
    if stockout_risk == "high":
        recommendation = "High stock-out risk. Reorder immediately."
    elif stockout_risk == "medium":
        recommendation = "Stock level is close to the reorder point. Reorder soon."
    else:
        recommendation = "Stock level looks sufficient for the next 7 open days."
        
    return {
        "total_predicted_sales": total_predicted_sales,
        "average_predicted_sales": average_predicted_sales,
        "expected_revenue": expected_revenue,
        "current_stock": current_stock,
        "projected_stock_after_7_days": projected_stock_after_7_days,
        "stockout_risk": stockout_risk,
        "recommended_reorder_quantity": recommended_reorder_quantity,
        "reorder_needed": reorder_needed,
        "recommendation": recommendation
    }
