import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\hamsa\Downloads\phase_2 detection\data\test.csv')

print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"  [{i}] {col}")

print(f"\nLabel column — unique values:")
# Try common label column names
for possible in ['label', 'Label', 'class', 'Class', 'attack_type', 'target']:
    if possible in df.columns:
        print(f"  Found '{possible}': {df[possible].unique()}")
        print(f"  Value counts:\n{df[possible].value_counts()}")
        break

print(f"\nFirst 3 rows:")
print(df.head(3))

print(f"\nCheck key filter columns exist:")
for col in ['Rate', 'syn_flag_number', 'Header_Length', 
            'rst_flag_number', 'UDP', 'ICMP', 'TCP']:
    exists = col in df.columns
    print(f"  {col}: {'YES' if exists else 'MISSING'}")