import pandas as pd
import numpy as np

print('Loading test.csv sections...')

# Load from multiple sections to get variety
df1 = pd.read_csv('data/test.csv', nrows=100000)
df2 = pd.read_csv('data/test.csv', skiprows=range(1, 200000), nrows=100000)
df3 = pd.read_csv('data/test.csv', skiprows=range(1, 500000), nrows=100000)
df4 = pd.read_csv('data/test.csv', skiprows=range(1, 800000), nrows=100000)

combined = pd.concat([df1, df2, df3, df4], ignore_index=True)
print(f'Total loaded: {len(combined)} rows')
print('\nAll available labels:')
print(combined['label'].value_counts())

# Define label groups
HIGH_RATE = [
    'DDoS-UDP_Flood', 'DDoS-SYN_Flood', 'DDoS-ICMP_Flood',
    'DDoS-TCP_Flood', 'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood',
    'DDoS-SynonymousIP_Flood', 'DDoS-ICMP_Fragmentation',
    'DDoS-UDP_Fragmentation', 'DDoS-ACK_Fragmentation',
    'Mirai-udpplain', 'Mirai-greip_flood', 'Mirai-greeth_flood',
    'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-SYN_Flood'
]

LOW_RATE = [
    'DoS-HTTP_Flood', 'DDoS-HTTP_Flood',
    'DDoS-SlowLoris', 'DNS-Flood'
]

BENIGN = ['BenignTraffic']

# Sample 100 rows per label — balanced
SAMPLES_PER_LABEL = 100
frames = []

print('\nSampling per label...')
for label in combined['label'].unique():
    subset = combined[combined['label'] == label]
    n      = min(SAMPLES_PER_LABEL, len(subset))
    sample = subset.sample(n=n, random_state=42)
    frames.append(sample)
    print(f'  {label:<35} : {n} rows')

balanced = pd.concat(frames, ignore_index=True)

# Shuffle
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f'\nFinal balanced dataset: {len(balanced)} rows')
print('\nFinal distribution:')
print(balanced['label'].value_counts())

# Save
balanced.to_csv('data/demo_balanced.csv', index=False)
print('\nSaved to data/demo_balanced.csv')