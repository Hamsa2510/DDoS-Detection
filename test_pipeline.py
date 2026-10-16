import pandas as pd
import numpy as np
from pipeline import DetectionPipeline

pipe = DetectionPipeline(
    weights_path='models/model_v2.weights.h5',
    scaler_path='models/minmax_scaler_v2.pkl'
)

df = pd.read_csv('data/test.csv')
print(f'Loaded {len(df)} rows from test.csv\n')

print('=' * 60)
print('PIPELINE TEST — 5 windows')
print('=' * 60)

for i in range(5):
    start = i * 10
    rows  = df.iloc[start: start + 10]

    if len(rows) < 10:
        break

    flow_dict  = rows.iloc[-1].to_dict()
    window_arr = pipe.model.prepare_window_from_df(rows)
    source_id  = f'src_{i:03d}'

    result = pipe.process_flow(flow_dict, window_arr, source_id)

    print(f'\nWindow {i+1} (source: {source_id})')
    for stage in result['stages']:
        print(f"  [{stage['stage']}] -> {stage['decision']}", end='')
        if 'reason' in stage:
            print(f"  | {stage['reason']}", end='')
        if 'confidence' in stage:
            print(f"  | conf: {stage['confidence']}", end='')
        print()
    print(f"  FINAL: {result['final_decision']} -- {result['final_reason']}")

print('\n' + '=' * 60)
print('STATS')
print('=' * 60)
for k, v in pipe.get_stats().items():
    print(f'  {k:<20}: {v}')