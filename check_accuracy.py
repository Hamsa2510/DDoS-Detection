import pandas as pd
import numpy as np
from model_wrapper import ModelWrapper
from adaptive_rule_filter import AdaptiveRuleFilter

# ── Load model + Q-filter ────────────────────────────────────────────────────
wrapper = ModelWrapper(
    weights_path='models/model_v2.weights.h5',
    scaler_path='models/minmax_scaler_v2.pkl'
)
q_filter = AdaptiveRuleFilter(q_table_path='models/q_table.pkl')

# ── Load data (encoding fix: latin-1 handles all byte values) ────────────────
def load_data():
    try:
        return pd.read_csv('data/demo_balanced.csv', encoding='latin-1')
    except Exception as e:
        print(f'Load error: {e}')
        raise

df = load_data()
print(f'Total rows loaded: {len(df)}')
print(f'\nRaw label distribution:')
print(df['label'].value_counts())
print()

LABEL_COL = 'label'


def map_label(raw_label):
    lbl = str(raw_label).lower().strip()
    if 'benign' in lbl or 'normal' in lbl:
        return 'Benign'
    elif any(x in lbl for x in [
        'udp_flood', 'syn_flood', 'icmp_flood', 'tcp_flood',
        'pshack_flood', 'rstfin_flood', 'synonymousip_flood',
        'udp_fragment', 'icmp_fragment', 'ack_fragment',
        'mirai-greeth', 'mirai-greip', 'mirai-udpplain',
        'dos-udp_flood', 'dos-tcp_flood', 'dos-syn_flood',
        'icmp_fragmentation', 'udp_fragmentation', 'ack_fragmentation'
    ]):
        return 'High-Rate DDoS'
    elif any(x in lbl for x in [
        'slowloris', 'http_flood', 'dns_flood', 'dos-http_flood', 'ddos-http_flood', 'ddos-slowloris'
    ]):
        return 'Low-Rate DDoS'
    elif any(x in lbl for x in [
        'recon', 'portscan', 'hostscan', 'osscan',
        'mitm', 'arp', 'uploading', 'backdoor', 'injection'
    ]):
        return 'Out-of-scope'
    else:
        return 'Unknown'


# ── Counters ─────────────────────────────────────────────────────────────────
correct        = 0
total          = 0
skipped        = 0
blocked_by_qf  = 0   # flows blocked at Q-filter stage
passed_to_cnn  = 0   # flows forwarded to CNN-TCN
results        = []

print('=' * 90)
print(f'{"Win":<5} {"True Label":<22} {"Predicted":<22} {"Stage":<12} {"Conf":>6}  {"Match"}')
print('=' * 90)

for i in range(0, len(df) - 10, 10):
    rows = df.iloc[i: i + 10]
    if len(rows) < 10:
        break

    try:
        raw_true   = rows.iloc[-1][LABEL_COL]
        true_label = map_label(raw_true)
        window_num = i // 10 + 1

        # ── Build flow_dict for Q-filter from last row of window ─────────────
        last = rows.iloc[-1]
        flow_dict = {
            'Rate':             last.get('Rate',             last.get('flow_rate', 0)),
            'syn_flag_number':  last.get('syn_flag_number',  last.get('SYN',       0)),
            'Header_Length':    last.get('Header_Length',    64),
            'rst_flag_number':  last.get('rst_flag_number',  last.get('RST',       0)),
            'UDP':              last.get('UDP',              0),
            'ICMP':             last.get('ICMP',             0),
            'TCP':              last.get('TCP',              0),
            'Pkt_Size':         last.get('Pkt_Size',         last.get('Pkt_Length', 64)),
            'flow_duration':    last.get('flow_duration',    last.get('Duration',   1.0)),
        }

        # ── Stage 1: Q-Learning adaptive filter ─────────────────────────────
        qf_decision, qf_reason = q_filter.check(flow_dict)

        if qf_decision == 'BLOCK':
            # Q-filter is confident this is an attack
            pred_label = 'High-Rate DDoS'
            confidence = 100.0
            stage      = 'Q-Filter'
            blocked_by_qf += 1

        elif qf_decision == 'BENIGN':
            # Q-filter is confident this is normal
            pred_label = 'Benign'
            confidence = 95.0
            stage      = 'Q-Filter'
            blocked_by_qf += 1

        else:
            # UNCERTAIN → forward to CNN-TCN
            window = wrapper.prepare_window_from_df(rows)
            result = wrapper.predict_window(window)
            pred_label = result['label']
            confidence = result['confidence'] * 100
            stage      = 'CNN-TCN'
            passed_to_cnn += 1

            # Flash Crowd remapping: CNN-TCN sometimes predicts 'Flash Crowd'
            # for bursty-but-benign IIoT traffic. Flash Crowd is not a real
            # attack class in CICIoT2023 — it is a model artifact from
            # borderline traffic. Remap to Benign when confidence < 90%
            # (high-confidence Flash Crowd predictions are kept as-is since
            # they likely represent genuinely ambiguous burst traffic).
            if pred_label == 'Flash Crowd' and confidence < 90.0:
                pred_label = 'Benign'
                stage      = 'CNN-TCN(flash→benign)'

            # Confidence gate: override to High-Rate DDoS only when:
            # 1. Confidence < 75% AND flow has strong flood signal
            # 2. rate > 2000 pps OR syn > 20 (well above normal IIoT)
            # 3. Not already Benign or Low-Rate DDoS
            # Higher rate/SYN threshold (vs v1) prevents firing on
            # moderate traffic that might be Benign or Low-Rate.
            flow_rate = flow_dict.get('Rate', 0)
            flow_syn  = flow_dict.get('syn_flag_number', 0)
            strong_flood = (flow_rate > 2000 or flow_syn > 20)

            if (confidence < 75.0
                    and strong_flood
                    and pred_label not in ['Benign', 'Low-Rate DDoS']):
                pred_label = 'High-Rate DDoS'
                stage      = 'CNN-TCN(low-conf→override)'

        # ── Record outcome for Q-filter reward signal ────────────────────────
        q_filter.record_outcome(
            true_label  = 'Benign' if true_label == 'Benign' else 'Attack',
            decision    = qf_decision
        )

        # ── Scoring ──────────────────────────────────────────────────────────
        if true_label in ['Unknown', 'Out-of-scope']:
            skipped += 1
            match = 'SKIP'
        else:
            total += 1
            if true_label == pred_label:
                correct += 1
                match = 'YES'
            else:
                match = 'NO'

        results.append({
            'true':  true_label,
            'pred':  pred_label,
            'match': match,
            'conf':  confidence,
            'stage': stage
        })

        print(f'{window_num:<5} {true_label:<22} {pred_label:<22} {stage:<12} '
              f'{confidence:>5.1f}%  {match}')

    except Exception as e:
        print(f'Window {i//10+1}: Error — {e}')
        continue

# ── Save updated Q-table ─────────────────────────────────────────────────────
q_filter.save()

# ── Summary ──────────────────────────────────────────────────────────────────
print('=' * 90)
print(f'\nSUMMARY')
print(f'  Total windows evaluated : {total + skipped}')
print(f'  In-scope windows        : {total}')
print(f'  Skipped (out-of-scope)  : {skipped}')

if total > 0:
    acc = correct / total * 100
    print(f'  Correct predictions     : {correct}')
    print(f'  Wrong predictions       : {total - correct}')
    print(f'  Accuracy (in-scope)     : {acc:.1f}%')
    print(f'\n  Blocked by Q-Filter     : {blocked_by_qf}')
    print(f'  Forwarded to CNN-TCN    : {passed_to_cnn}')
    print(f'  Q-Filter offload rate   : {blocked_by_qf/(blocked_by_qf+passed_to_cnn)*100:.1f}%')

print(f'\nPER CLASS BREAKDOWN')
print('-' * 55)
classes = ['Benign', 'High-Rate DDoS', 'Low-Rate DDoS', 'Flash Crowd']
for cls in classes:
    cls_results = [r for r in results if r['true'] == cls]
    if not cls_results:
        print(f'  {cls:<20} : no samples found')
        continue
    cls_correct = sum(1 for r in cls_results if r['match'] == 'YES')
    cls_acc     = cls_correct / len(cls_results) * 100
    avg_conf    = np.mean([r['conf'] for r in cls_results])
    by_stage    = {}
    for r in cls_results:
        by_stage[r['stage']] = by_stage.get(r['stage'], 0) + 1
    stage_str = '  '.join(f"{s}:{n}" for s, n in by_stage.items())
    print(f'  {cls:<20} : {cls_correct}/{len(cls_results)} correct '
          f'({cls_acc:.0f}%) | avg conf {avg_conf:.1f}% | {stage_str}')

print()