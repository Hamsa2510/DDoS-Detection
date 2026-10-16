import pandas as pd
import numpy as np
import sys
sys.path.append(r'C:\Users\hamsa\Downloads\phase_2 detection')
from adaptive_rule_filter import AdaptiveRuleFilter

# ── Label mapping to your 4-class schema ────────────────────────
HIGH_RATE = {
    'DDoS-SynonymousIP_Flood','DDoS-TCP_Flood','DDoS-UDP_Flood',
    'DDoS-PSHACK_Flood','DDoS-SYN_Flood','DDoS-RSTFINFlood',
    'DDoS-ICMP_Flood','DoS-UDP_Flood','DoS-TCP_Flood','DoS-SYN_Flood',
    'DDoS-ICMP_Fragmentation','DDoS-ACK_Fragmentation',
    'DDoS-UDP_Fragmentation','Mirai-greeth_flood',
    'Mirai-greip_flood','Mirai-udpplain'
}
LOW_RATE = {
    'DDoS-SlowLoris','DDoS-HTTP_Flood','DoS-HTTP_Flood',
    'DNS_Spoofing','MITM-ArpSpoofing','Recon-HostDiscovery',
    'Recon-OSScan','Recon-PortScan','Recon-PingSweep',
    'VulnerabilityScan','DictionaryBruteForce','SqlInjection',
    'BrowserHijacking','CommandInjection','XSS',
    'Backdoor_Malware','Uploading_Attack'
}

def map_label(raw_label):
    if raw_label == 'BenignTraffic':
        return 'Benign'
    elif raw_label in HIGH_RATE:
        return 'High-Rate'
    elif raw_label in LOW_RATE:
        return 'Low-Rate'
    else:
        return 'Benign'

# ── Load data ────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    r'C:\Users\hamsa\Downloads\phase_2 detection\data\test.csv'
)
df['mapped_label'] = df['label'].apply(map_label)

benign    = df[df['mapped_label'] == 'Benign']
high_rate = df[df['mapped_label'] == 'High-Rate']
low_rate  = df[df['mapped_label'] == 'Low-Rate']

# Drifting stream: normal → attack → normal → low-rate → high-rate
# This simulates real IIoT traffic drift to test Q-Learning adaptation
df_sample = pd.concat([
    benign.sample(5000,     random_state=42),   # phase 1: normal
    high_rate.sample(20000, random_state=42),   # phase 2: high-rate flood
    benign.sample(3000,     random_state=1),    # phase 3: returns to normal
    low_rate,                                   # phase 4: low-rate attack
    high_rate.sample(20000, random_state=2)     # phase 5: high-rate resumes
]).reset_index(drop=True)

print(f"Drifting stream: {len(df_sample)} rows")
print(df_sample['mapped_label'].value_counts())

# ── Ablation loop ────────────────────────────────────────────────
window_sizes = [25, 50, 100, 200]
results      = []

for W in window_sizes:
    print(f"\nTesting W={W} ...")

    filt = AdaptiveRuleFilter(
        q_table_path=r'C:\Users\hamsa\Downloads\phase_2 detection\models\q_table_new.pkl'
    )
    filt.update_interval = W

    total_flows     = 0
    total_blocked   = 0
    correct_blocked = 0
    false_positives = 0
    total_benign    = 0
    total_attacks   = 0

    for _, row in df_sample.iterrows():
        flow_dict = {
            'Rate':            row['Rate'],
            'syn_flag_number': row['syn_flag_number'],
            'Header_Length':   row['Header_Length'],
            'rst_flag_number': row['rst_flag_number'],
            'UDP':             row['UDP'],
            'ICMP':            row['ICMP'],
            'TCP':             row['TCP'],
            'flow_duration':   row['flow_duration'],
            'Pkt_Size':        row['Variance'],
        }
        true_label = row['mapped_label']
        is_attack  = true_label != 'Benign'

        decision, _ = filt.check(flow_dict)
        filt.record_outcome(true_label, decision)

        is_blocked = (decision == 'BLOCK')

        total_flows += 1
        if is_attack:
            total_attacks += 1
        else:
            total_benign += 1

        if is_blocked:
            total_blocked += 1
            if is_attack:
                correct_blocked += 1
            else:
                false_positives += 1

    precision      = correct_blocked / total_blocked if total_blocked > 0 else 0
    fpr            = false_positives / total_benign  if total_benign  > 0 else 0
    load_reduction = total_blocked   / total_flows   if total_flows   > 0 else 0
    recall         = correct_blocked / total_attacks if total_attacks > 0 else 0

    results.append({
        'Window Size (W)':               W,
        'Pre-filter Precision (%)':      round(precision      * 100, 2),
        'False Positive Rate (%)':       round(fpr            * 100, 2),
        'Classifier Load Reduction (%)': round(load_reduction * 100, 2),
        'Attack Recall (%)':             round(recall         * 100, 2),
    })
    print(f"  Precision={precision*100:.2f}%  FPR={fpr*100:.2f}%  "
          f"Load Reduction={load_reduction*100:.2f}%  Recall={recall*100:.2f}%")

# ── Print and save ───────────────────────────────────────────────
df_res = pd.DataFrame(results)
print("\n" + "="*70)
print("WINDOW SIZE ABLATION RESULTS — DRIFTING STREAM")
print("="*70)
print(df_res.to_string(index=False))
print("="*70)

df_res.to_csv(
    r'C:\Users\hamsa\Downloads\phase_2 detection\window_ablation_results.csv',
    index=False
)
print("\nSaved to window_ablation_results.csv")