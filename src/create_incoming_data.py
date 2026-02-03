import pandas as pd

# 1. Load yesterday's data (READ-ONLY)
df = pd.read_csv("data/raw/Mall_Customers.csv")

# 2. Create a copy (never modify raw data)
new_df = df.copy()

# 3. Simulate real-world changes
# Younger customers
new_df['Age'] = new_df['Age'] - 10

# Higher spending behavior
new_df['Spending Score (1-100)'] = new_df['Spending Score (1-100)'] + 20

# 4. Save as incoming data (today's data)
new_df.to_csv("data/incoming/Customers_day2.csv", index=False)

print("Incoming data created successfully")
