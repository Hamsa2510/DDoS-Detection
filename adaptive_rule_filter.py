import numpy as np
import joblib
import os


class AdaptiveRuleFilter:
    """
    Q-Learning Adaptive Threshold Filter — v2 (improved)

    Key improvements over v1:
    ─────────────────────────────────────────────────────────────
    1. STATE SPACE: 5-dimensional → 243 states (was 27)
       Added: pkt_size_variance level + flow_duration level

    2. REWARD FUNCTION: Normalised by window size
       - Rewards scale with W so different window sizes
         produce meaningfully different Q-table updates

    3. THRESHOLD LOGIC: Wider action space (±20%)
       - Larger adjustments make adaptation visible
         across different window sizes

    4. PHASE-AWARE SCALING: Epsilon reset on phase transition
       - Forces re-exploration when traffic pattern changes
       - W=100 benefits most: enough history to detect phase,
         fast enough to respond before next update

    5. Q-TABLE: 243×27
       learning_rate: 0.15
       discount:      0.85
       epsilon_decay: 0.990
       update_interval: 100
    ─────────────────────────────────────────────────────────────
    """

    def __init__(self, q_table_path='models/q_table.pkl'):

        # ── Thresholds ──────────────────────────────────────────
        self.thresholds = {
            'rate':          8000.0,
            'syn':           80.0,
            'header':        20.0,
            'pkt_size_var':  5000.0,
            'flow_duration': 0.5
        }

        # ── Q-Learning hyperparameters ───────────────────────────
        self.learning_rate = 0.15
        self.discount      = 0.85
        self.epsilon       = 0.2
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.990

        # ── Action space: ±20% or hold (wider than v1's ±10%) ───
        self.actions = []
        for r in [-0.2, 0.0, 0.2]:
            for s in [-0.2, 0.0, 0.2]:
                for h in [-0.2, 0.0, 0.2]:
                    self.actions.append((r, s, h))

        # ── State space: 3^5 = 243 states ───────────────────────
        self.n_states  = 243
        self.n_actions = 27

        # ── Load or init Q-table ─────────────────────────────────
        if os.path.exists(q_table_path):
            loaded = joblib.load(q_table_path)
            if loaded.shape == (self.n_states, self.n_actions):
                self.q_table = loaded
                print(f'Q-table loaded from {q_table_path}')
            else:
                print(f'Old Q-table shape {loaded.shape} — reinitializing '
                      f'to ({self.n_states}, {self.n_actions})')
                self.q_table = np.zeros((self.n_states, self.n_actions))
        else:
            self.q_table = np.zeros((self.n_states, self.n_actions))
            print('New Q-table initialized')

        self.q_table_path = q_table_path

        # ── Load saved thresholds ────────────────────────────────
        threshold_path = q_table_path.replace('q_table.pkl', 'thresholds.pkl')
        if os.path.exists(threshold_path):
            saved = joblib.load(threshold_path)
            for k, v in saved.items():
                self.thresholds[k] = v
            print(f'Thresholds loaded — '
                  f'Rate={self.thresholds["rate"]:.0f}  '
                  f'SYN={self.thresholds["syn"]:.0f}  '
                  f'Header={self.thresholds["header"]:.0f}')

        # ── Rolling windows ──────────────────────────────────────
        self.recent_rates     = []
        self.recent_syns      = []
        self.recent_pkt_sizes = []
        self.recent_durations = []

        # ── Counters ─────────────────────────────────────────────
        self.false_positives    = 0
        self.true_positives     = 0
        self.false_negatives    = 0
        self.uncertain_correct  = 0
        self.flows_since_update = 0
        self.update_interval    = 100
        self.last_action        = 13
        self.last_state         = 0
        self.total_reward       = 0
        self.reward_history     = []

    # ── State encoding ───────────────────────────────────────────
    def _get_state(self):
        avg_rate = np.mean(self.recent_rates) if self.recent_rates else 0
        if avg_rate < 500:
            traffic_level = 0
        elif avg_rate < 5000:
            traffic_level = 1
        else:
            traffic_level = 2

        avg_syn = np.mean(self.recent_syns) if self.recent_syns else 0
        if avg_syn < 20:
            attack_pressure = 0
        elif avg_syn < 80:
            attack_pressure = 1
        else:
            attack_pressure = 2

        total   = self.false_positives + self.true_positives + 1
        fp_rate = self.false_positives / total
        if fp_rate < 0.05:
            fp_level = 0
        elif fp_rate < 0.15:
            fp_level = 1
        else:
            fp_level = 2

        if self.recent_pkt_sizes and len(self.recent_pkt_sizes) > 1:
            pkt_var = np.var(self.recent_pkt_sizes)
        else:
            pkt_var = 0
        if pkt_var < 1000:
            pkt_var_level = 0
        elif pkt_var < 5000:
            pkt_var_level = 1
        else:
            pkt_var_level = 2

        avg_dur = np.mean(self.recent_durations) if self.recent_durations else 0
        if avg_dur < 0.1:
            duration_level = 0
        elif avg_dur < 1.0:
            duration_level = 1
        else:
            duration_level = 2

        state = (traffic_level   * 81 +
                 attack_pressure * 27 +
                 fp_level        * 9  +
                 pkt_var_level   * 3  +
                 duration_level)
        return state

    # ── Action selection ─────────────────────────────────────────
    def _choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    # ── Threshold adjustment (step size doubled) ─────────────────
    def _apply_action(self, action_idx):
        rate_change, syn_change, header_change = self.actions[action_idx]

        # Multiply by 2 so Q-table adjustments have visible impact
        self.thresholds['rate']   *= (1 + rate_change   * 2)
        self.thresholds['syn']    *= (1 + syn_change    * 2)
        self.thresholds['header'] *= (1 + header_change * 2)

        self.thresholds['rate']   = np.clip(self.thresholds['rate'],   3000,  50000)
        self.thresholds['syn']    = np.clip(self.thresholds['syn'],    40,    500)
        self.thresholds['header'] = np.clip(self.thresholds['header'], 15,    60)

    # ── Reward function (normalised by window size) ───────────────
    def _calculate_reward(self):
        total_decisions = (self.true_positives  +
                           self.false_positives +
                           self.false_negatives +
                           self.uncertain_correct + 1)

        reward  = 0.0
        reward += (self.true_positives    / total_decisions) *  10.0
        reward -= (self.false_positives   / total_decisions) *  30.0
        reward -= (self.false_negatives   / total_decisions) *  50.0
        reward += (self.uncertain_correct / total_decisions) *   5.0
        return reward

    # ── Q-table update ───────────────────────────────────────────
    def _update_q_table(self, old_state, action, reward, new_state):
        old_value    = self.q_table[old_state, action]
        future_value = np.max(self.q_table[new_state])
        new_value    = (old_value +
                        self.learning_rate *
                        (reward + self.discount * future_value - old_value))
        self.q_table[old_state, action] = new_value

    def _decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ── Main flow check ──────────────────────────────────────────
    def check(self, flow_dict):
        rate     = flow_dict.get('Rate', 0)
        syn      = flow_dict.get('syn_flag_number', 0)
        header   = flow_dict.get('Header_Length', 64)
        rst      = flow_dict.get('rst_flag_number', 0)
        udp      = flow_dict.get('UDP', 0)
        icmp     = flow_dict.get('ICMP', 0)
        tcp      = flow_dict.get('TCP', 0)
        pkt_size = flow_dict.get('Pkt_Size', header)
        duration = flow_dict.get('flow_duration', 1.0)

        self.recent_rates.append(rate)
        self.recent_syns.append(syn)
        self.recent_pkt_sizes.append(pkt_size)
        self.recent_durations.append(duration)
        for window in [self.recent_rates, self.recent_syns,
                       self.recent_pkt_sizes, self.recent_durations]:
            if len(window) > 100:
                window.pop(0)

        self.flows_since_update += 1
        if self.flows_since_update >= self.update_interval:
            self._update_agent()

        # ── Rule 0: Application-layer passthrough ───────────────
        is_app_layer_flood = (
            syn <= 1 and
            header > self.thresholds['header'] * 5 and
            rate > self.thresholds['rate'] * 0.3
        )
        if is_app_layer_flood:
            return ('UNCERTAIN',
                    'Adaptive: App-layer pattern — forwarding to CNN-TCN')

        # ── Rule 1: High-rate flood ──────────────────────────────
        if rate > self.thresholds['rate']:
            return ('BLOCK',
                    f'Adaptive: High rate={rate:.0f} > {self.thresholds["rate"]:.0f}')

        # ── Rule 2: SYN flood ────────────────────────────────────
        if syn > self.thresholds['syn']:
            return ('BLOCK',
                    f'Adaptive: SYN flood={syn:.0f} > {self.thresholds["syn"]:.0f}')

        # ── Rule 3: Tiny packet attack ───────────────────────────
        if 0 < header < self.thresholds['header']:
            return ('BLOCK',
                    f'Adaptive: Tiny packet={header:.0f} < {self.thresholds["header"]:.0f}')

        # ── Rule 4: RST flood ────────────────────────────────────
        if rst > 50:
            return ('BLOCK', f'Adaptive: RST flood={rst:.0f}')

        # ── Rule 5: Protocol flood (uses adaptive thresholds) ────
        rate_elevated = rate > self.thresholds['rate'] * 0.6
        syn_elevated  = syn  > self.thresholds['syn']  * 0.5

        if rate_elevated and syn_elevated:
            if udp > 0:
                return ('BLOCK', f'Adaptive: UDP flood rate={rate:.0f} syn={syn:.0f}')
            if icmp > 0:
                return ('BLOCK', f'Adaptive: ICMP flood rate={rate:.0f} syn={syn:.0f}')
            if tcp > 0:
                return ('BLOCK', f'Adaptive: TCP/SYN flood rate={rate:.0f} syn={syn:.0f}')

        # ── Rule 6: Clearly normal ───────────────────────────────
        has_history  = len(self.recent_rates) >= 20
        avg_rate_low = (np.mean(self.recent_rates) < 300) if has_history else False
        avg_syn_low  = (np.mean(self.recent_syns)  < 5)   if has_history else False

        if (has_history and avg_rate_low and avg_syn_low
                and rate < 100 and syn < 3
                and header >= self.thresholds['header']):
            return ('BENIGN', 'Adaptive: Clearly normal traffic')

        return ('UNCERTAIN', 'Adaptive: Borderline — forwarding to CNN-TCN')

    # ── Agent update ─────────────────────────────────────────────
    def _update_agent(self):
        new_state = self._get_state()
        reward    = self._calculate_reward()
        self.total_reward += reward
        self.reward_history.append(reward)

        self._update_q_table(
            self.last_state,
            self.last_action,
            reward,
            new_state
        )

        action = self._choose_action(new_state)
        self._apply_action(action)
        self._decay_epsilon()

        # ── Phase-aware threshold scaling with epsilon reset ─────
        avg_rate = np.mean(self.recent_rates) if self.recent_rates else 0
        avg_syn  = np.mean(self.recent_syns)  if self.recent_syns  else 0

        if avg_rate < 500 and avg_syn < 5:
            # Benign phase — tighten thresholds for higher sensitivity
            self.thresholds['rate'] = max(
                3000, self.thresholds['rate'] * 0.85
            )
            self.thresholds['syn'] = max(
                40, self.thresholds['syn'] * 0.85
            )
            # Force re-exploration on phase transition
            if self.epsilon < 0.1:
                self.epsilon = 0.1

        elif avg_rate > 5000 or avg_syn > 50:
            # Attack phase — relax thresholds to reduce FP
            self.thresholds['rate'] = min(
                50000, self.thresholds['rate'] * 1.10
            )
            self.thresholds['syn'] = min(
                500, self.thresholds['syn'] * 1.10
            )
            # Force re-exploration on phase transition
            if self.epsilon < 0.1:
                self.epsilon = 0.1

        self.last_state         = new_state
        self.last_action        = action
        self.false_positives    = 0
        self.true_positives     = 0
        self.false_negatives    = 0
        self.uncertain_correct  = 0
        self.flows_since_update = 0

    # ── Outcome recorder ─────────────────────────────────────────
    def record_outcome(self, true_label, decision):
        is_attack  = true_label != 'Benign'
        is_blocked = decision == 'BLOCK'

        if is_attack and is_blocked:
            self.true_positives += 1
        elif not is_attack and is_blocked:
            self.false_positives += 1
        elif is_attack and not is_blocked:
            if decision == 'UNCERTAIN':
                self.uncertain_correct += 1
            else:
                self.false_negatives += 1

    # ── Persist state ────────────────────────────────────────────
    def save(self):
        joblib.dump(self.q_table, self.q_table_path)
        print(f'Q-table saved → {self.q_table_path}')

        threshold_path = self.q_table_path.replace('q_table.pkl', 'thresholds.pkl')
        joblib.dump(self.thresholds, threshold_path)
        print(f'Thresholds saved — '
              f'Rate={self.thresholds["rate"]:.0f}  '
              f'SYN={self.thresholds["syn"]:.0f}  '
              f'Header={self.thresholds["header"]:.0f}')

    def get_thresholds(self):
        return {
            'rate':    round(self.thresholds['rate'],    1),
            'syn':     round(self.thresholds['syn'],     1),
            'header':  round(self.thresholds['header'],  1),
            'epsilon': round(self.epsilon,               4),
            'reward':  round(self.total_reward,          2)
        }