# ================================================================
# rule_filter.py
# Fast rule-based filter — runs BEFORE the CNN-TCN model
# Catches obvious attacks instantly without needing ML
# ================================================================

class RuleFilter:
    """
    Checks a single traffic flow against hard rules.
    Returns:
        'BLOCK'     — obvious attack, block immediately
        'UNCERTAIN' — not obvious, send to CNN-TCN model
        'BENIGN'    — clearly normal, skip model (fast path)
    """

    def __init__(self):
        # Thresholds — based on CICIoT2023 feature ranges
        self.RATE_HIGH       = 10000   # packets/sec — clear flood
        self.RATE_LOW        = 0.01    # suspiciously slow
        self.SYN_FLOOD       = 100     # syn_flag_number threshold
        self.TINY_PACKET     = 20      # Header_Length in bytes
        self.RST_FLOOD       = 50      # rst_flag_number threshold

    def check(self, flow: dict) -> tuple:
        """
        flow: dict with feature names as keys
        Returns: (decision, reason)
        """
        rate   = flow.get('Rate', 0)
        syn    = flow.get('syn_flag_number', 0)
        header = flow.get('Header_Length', 64)
        rst    = flow.get('rst_flag_number', 0)

        # Rule 1 — extremely high packet rate = flood attack
        if rate > self.RATE_HIGH:
            return ('BLOCK', f'High rate flood: Rate={rate:.1f}')

        # Rule 2 — SYN flood pattern
        if syn > self.SYN_FLOOD:
            return ('BLOCK', f'SYN flood: syn_flag={syn}')

        # Rule 3 — tiny packets = fragmentation/low-rate attack
        if 0 < header < self.TINY_PACKET:
            return ('BLOCK', f'Tiny packets: Header_Length={header}')

        # Rule 4 — RST flood
        if rst > self.RST_FLOOD:
            return ('BLOCK', f'RST flood: rst_flag={rst}')

        # Rule 5 — clearly normal (very low rate, normal header)
        if rate < 100 and syn < 5 and rst < 5:
            return ('BENIGN', 'Passes all rule checks')

        # Everything else — uncertain, let model decide
        return ('UNCERTAIN', 'Borderline — forwarding to CNN-TCN')


# ── Quick test ──────────────────────────────────────────────────
if __name__ == '__main__':
    rf = RuleFilter()

    test_flows = [
        {'Rate': 50000, 'syn_flag_number': 0,   'Header_Length': 64, 'rst_flag_number': 0},
        {'Rate': 200,   'syn_flag_number': 200,  'Header_Length': 64, 'rst_flag_number': 0},
        {'Rate': 50,    'syn_flag_number': 2,    'Header_Length': 10, 'rst_flag_number': 0},
        {'Rate': 30,    'syn_flag_number': 1,    'Header_Length': 64, 'rst_flag_number': 0},
        {'Rate': 500,   'syn_flag_number': 10,   'Header_Length': 64, 'rst_flag_number': 5},
    ]

    print('Rule Filter Test')
    print('=' * 55)
    for i, flow in enumerate(test_flows):
        decision, reason = rf.check(flow)
        print(f'Flow {i+1}: [{decision:10s}] {reason}')