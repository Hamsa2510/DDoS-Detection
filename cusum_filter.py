# ================================================================
# cusum_filter.py
# CUSUM (Cumulative Sum) filter — runs AFTER model says "Benign"
# Catches bots that send individually-normal traffic but slowly
# escalate over time — exactly the scenario your professor raised
# ================================================================

import time
from collections import defaultdict


class CUSUMFilter:
    """
    Tracks cumulative drift in Rate and Number features per source.

    How CUSUM works:
    - For each source, keep a running cumulative sum (S)
    - Every new flow: S = max(0, S + (value - target) - slack)
    - If S exceeds threshold → source is drifting upward → SUSPICIOUS

    Think of it as: normal traffic fluctuates around a baseline.
    A bot slowly pushes above that baseline. CUSUM detects the drift
    even when no single packet is obviously bad.
    """

    def __init__(self):
        # CUSUM parameters
        # target = expected normal value (based on CICIoT2023 benign stats)
        # slack  = how much deviation is acceptable before accumulating
        # threshold = cumulative score that triggers alert

        self.RATE_TARGET    = 50.0    # expected normal Rate
        self.RATE_SLACK     = 30.0    # allowable fluctuation
        self.RATE_THRESHOLD = 500.0   # cumulative drift limit

        self.NUM_TARGET     = 10.0    # expected normal Number (packets)
        self.NUM_SLACK      = 8.0
        self.NUM_THRESHOLD  = 200.0

        self.COOLDOWN_SEC   = 60      # seconds before resetting a flagged source

        # Per-source state: {source_id: {'S_rate': float, 'S_num': float, 'flagged_at': time}}
        self._state = defaultdict(lambda: {
            'S_rate': 0.0,
            'S_num':  0.0,
            'flagged_at': None,
            'flow_count': 0
        })

    def check(self, source_id: str, rate: float, number: float) -> tuple:
        """
        source_id: string identifier for the traffic source
                   (in simulation, use row index as string e.g. 'src_001')
        rate:      Rate feature value (already scaled 0-1, we rescale internally)
        number:    Number feature value

        Returns: (decision, score_info)
            'ALLOW'      — cumulative drift is normal
            'SUSPICIOUS' — drift is rising, flag for review
        """
        state = self._state[source_id]

        # If flagged before, check cooldown
        if state['flagged_at'] is not None:
            elapsed = time.time() - state['flagged_at']
            if elapsed < self.COOLDOWN_SEC:
                return ('SUSPICIOUS', f'Still in cooldown ({elapsed:.0f}s / {self.COOLDOWN_SEC}s)')
            else:
                # Reset after cooldown
                state['S_rate'] = 0.0
                state['S_num']  = 0.0
                state['flagged_at'] = None

        # Update CUSUM scores
        # S = max(0, S_prev + (observed - target) - slack)
        state['S_rate'] = max(0.0, state['S_rate'] + (rate   - self.RATE_TARGET) - self.RATE_SLACK)
        state['S_num']  = max(0.0, state['S_num']  + (number - self.NUM_TARGET)  - self.NUM_SLACK)
        state['flow_count'] += 1

        score_info = (f'S_rate={state["S_rate"]:.1f} '
                      f'S_num={state["S_num"]:.1f} '
                      f'flows={state["flow_count"]}')

        # Check if either score exceeds threshold
        if state['S_rate'] > self.RATE_THRESHOLD or state['S_num'] > self.NUM_THRESHOLD:
            state['flagged_at'] = time.time()
            return ('SUSPICIOUS', f'Drift detected! {score_info}')

        return ('ALLOW', score_info)

    def reset_source(self, source_id: str):
        """Manually reset a source (e.g. after admin review)."""
        if source_id in self._state:
            del self._state[source_id]

    def get_all_scores(self) -> dict:
        """Return current CUSUM scores for all tracked sources."""
        return {
            sid: {'S_rate': s['S_rate'], 'S_num': s['S_num'],
                  'flows': s['flow_count'], 'flagged': s['flagged_at'] is not None}
            for sid, s in self._state.items()
        }


# ── Quick test ──────────────────────────────────────────────────
if __name__ == '__main__':
    cf = CUSUMFilter()

    print('CUSUM Filter Test')
    print('=' * 60)
    print('Simulating a bot that slowly increases Rate over 10 flows...')
    print()

    # Simulate normal source — stays flat
    print('[ Normal source ]')
    for i in range(8):
        rate = 40 + (i * 2)   # slight variation, stays low
        d, info = cf.check('normal_src', rate, 8)
        print(f'  Flow {i+1}: Rate={rate:.0f}  -> {d:12s}  {info}')

    print()

    # Simulate bot — rate slowly climbs
    print('[ Bot source — slowly escalating ]')
    for i in range(10):
        rate = 60 + (i * 25)   # climbs from 60 → 285
        num  = 10 + (i * 3)
        d, info = cf.check('bot_src', rate, num)
        print(f'  Flow {i+1}: Rate={rate:.0f}  -> {d:12s}  {info}')
        if d == 'SUSPICIOUS':
            print('  *** CUSUM triggered — bot detected! ***')
            break