import pandas as pd
from model_wrapper import ModelWrapper

wrapper = ModelWrapper(
    weights_path='models/model_v2.weights.h5',
    scaler_path='models/minmax_scaler_v2.pkl'
)

df     = pd.read_csv('data/demo_balanced.csv')
benign = df[df['label'] == 'BenignTraffic'].reset_index(drop=True)
print(f'Benign rows: {len(benign)}')
print()

correct_raw   = 0
correct_fixed = 0
total         = 0

print(f'{"Win":<5} {"Raw Label":<22} {"Conf":>6}  {"Fixed Label":<22} {"Match"}')
print('=' * 70)

for i in range(0, min(90, len(benign) - 10), 10):
    rows   = benign.iloc[i: i + 10]
    window = wrapper.prepare_window_from_df(rows)
    result = wrapper.predict_window(window)

    raw_label = result['label']
    raw_conf  = result['confidence']
    raw_id    = result['class_id']

    # Apply same fix as pipeline.py
    if raw_id == 3 and raw_conf < 0.75:
        fixed_label = 'Benign'
    else:
        fixed_label = raw_label

    total += 1
    if raw_label   == 'Benign': correct_raw   += 1
    if fixed_label == 'Benign': correct_fixed += 1

    fixed_match = 'YES' if fixed_label == 'Benign' else 'NO'

    print(f'{i//10+1:<5} {raw_label:<22} {raw_conf*100:>5.1f}%  '
          f'{fixed_label:<22} {fixed_match}')

print()
print(f'Raw accuracy   : {correct_raw}/{total}   '
      f'({correct_raw/total*100:.0f}%)')
print(f'Fixed accuracy : {correct_fixed}/{total}  '
      f'({correct_fixed/total*100:.0f}%)')
print()
print('Note: Flash Crowd < 75% confidence -> converted to Benign')
print('      Flash Crowd >= 75% confidence -> stays as Flash Crowd (BLOCK)')