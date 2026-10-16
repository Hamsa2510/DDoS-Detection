import pandas as pd
import numpy as np
from adaptive_rule_filter import AdaptiveRuleFilter

print('Loading data...')
df = pd.read_csv('data/demo_balanced.csv')
print(f'Loaded {len(df)} rows')

# ── Label mapping ──────────────────────────────────────────────
HIGH_RATE_LABELS = [
    'ddos-icmp_flood','ddos-udp_flood','ddos-syn_flood',
    'ddos-tcp_flood','ddos-pshack_flood','ddos-rstfinflood',
    'ddos-synonymousip_flood','ddos-icmp_fragmentation',
    'ddos-udp_fragmentation','ddos-ack_fragmentation',
    'mirai-udpplain','mirai-greip_flood','mirai-greeth_flood',
    'dos-udp_flood','dos-tcp_flood','dos-syn_flood'
]

def get_true_class(label):
    lbl = str(label).lower().strip()
    if 'benign' in lbl or 'normal' in lbl:
        return 'Benign'
    for pat in HIGH_RATE_LABELS:
        if pat in lbl:
            return 'HighRate'
    return 'LowRate'

# Add true class column
df['true_class'] = df['label'].apply(get_true_class)
print('\nTrue class distribution:')
print(df['true_class'].value_counts())

# ── Check Rate distribution per class ─────────────────────────
print('\nRate statistics per class:')
for cls in ['Benign', 'HighRate', 'LowRate']:
    subset = df[df['true_class'] == cls]['Rate']
    print(f'  {cls:<10} mean={subset.mean():.1f} '
          f'max={subset.max():.1f} '
          f'min={subset.min():.1f}')

# ── Create agent ───────────────────────────────────────────────
agent = AdaptiveRuleFilter(q_table_path='models/q_table.pkl')

agent.q_table = np.zeros((agent.n_states, agent.n_actions))
agent.epsilon = 0.5
agent.thresholds = {
    'rate':   10000.0,
    'syn':    100.0,
    'header': 20.0
}
agent.total_reward       = 0
agent.false_positives    = 0
agent.true_positives     = 0
agent.false_negatives    = 0
agent.flows_since_update = 0

N_EPISODES = 50
print(f'\nTraining for {N_EPISODES} episodes...')
print('=' * 75)

best_accuracy = 0
best_thresholds = None

for episode in range(N_EPISODES):
    episode_df = df.sample(frac=1, random_state=episode).reset_index(drop=True)

    tp = 0  # true positives  — attack correctly blocked
    tn = 0  # true negatives  — benign correctly passed
    fp = 0  # false positives — benign wrongly blocked
    fn = 0  # false negatives — attack wrongly passed
    total = 0

    for i, row in episode_df.iterrows():
        flow_dict  = row.to_dict()
        true_class = row['true_class']
        decision, reason = agent.check(flow_dict)

        total += 1

        if true_class == 'HighRate':
            if decision == 'BLOCK':
                # Correctly blocked obvious flood
                agent.true_positives += 1
                tp += 1
            else:
                # Sent to model — acceptable for borderline cases
                # Count as correct since model will handle it
                tn += 1

        elif true_class == 'LowRate':
            if decision == 'UNCERTAIN':
                # Correct — low rate needs model
                tn += 1
            elif decision == 'BENIGN':
                # Wrongly passed as benign without model check
                fn += 1
            elif decision == 'BLOCK':
                # Rule blocked a low-rate attack — good if Rate is high
                rate = flow_dict.get('Rate', 0)
                if rate > 2000:
                    agent.true_positives += 1
                    tp += 1
                else:
                    # False block on low-rate
                    agent.false_positives += 1
                    fp += 1

        elif true_class == 'Benign':
            if decision == 'BLOCK':
                # Wrongly blocked benign — bad
                agent.false_positives += 1
                fp += 1
            else:
                # Correctly passed benign
                tn += 1

    # Calculate metrics
    correct  = tp + tn
    accuracy = correct / total * 100
    fp_rate  = fp / max(total, 1) * 100

    thresholds = agent.get_thresholds()

    # Track best result
    if accuracy > best_accuracy:
        best_accuracy   = accuracy
        best_thresholds = thresholds.copy()

    print(f'Episode {episode+1:2d}/{N_EPISODES} | '
          f'Acc: {accuracy:.1f}% | '
          f'TP:{tp:4d} TN:{tn:4d} FP:{fp:3d} | '
          f'Rate:{thresholds["rate"]:7.0f} | '
          f'SYN:{thresholds["syn"]:6.1f} | '
          f'Reward:{thresholds["reward"]:8.0f}')

agent.save()

print('\n' + '=' * 75)
print('Training complete!')
print(f'\nBest accuracy achieved : {best_accuracy:.1f}%')
print(f'Best thresholds        :')
if best_thresholds:
    print(f'  Rate   : {best_thresholds["rate"]}')
    print(f'  SYN    : {best_thresholds["syn"]}')
    print(f'  Header : {best_thresholds["header"]}')

print(f'\nFinal thresholds:')
print(f'  Rate threshold   : {agent.thresholds["rate"]:.1f}')
print(f'  SYN threshold    : {agent.thresholds["syn"]:.1f}')
print(f'  Header threshold : {agent.thresholds["header"]:.1f}')