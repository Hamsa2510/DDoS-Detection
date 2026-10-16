from adaptive_rule_filter import AdaptiveRuleFilter
from cusum_filter import CUSUMFilter
from model_wrapper import ModelWrapper


class DetectionPipeline:

    def __init__(self, weights_path, scaler_path):
        self.rule_filter  = AdaptiveRuleFilter(
            q_table_path='models/q_table.pkl'
        )
        self.cusum_filter = CUSUMFilter()
        self.model        = ModelWrapper(weights_path, scaler_path)

        self.stats = {
            'total': 0,
            'blocked_rule': 0,
            'blocked_model': 0,
            'suspicious': 0,
            'allowed': 0
        }

    def process_flow(self, flow_dict, window_array=None,
                     source_id='unknown'):
        self.stats['total'] += 1
        result = {
            'source_id': source_id,
            'stages':    [],
            'subtype':   '--'
        }

        # STAGE 1 — Adaptive Rule Filter
        rule_decision, rule_reason = self.rule_filter.check(flow_dict)
        result['stages'].append({
            'stage':    'Adaptive Rule Filter',
            'decision': rule_decision,
            'reason':   rule_reason
        })

        if rule_decision == 'BLOCK':
            self.stats['blocked_rule'] += 1
            result['final_decision'] = 'BLOCK'
            result['final_reason']   = f'Rule: {rule_reason}'
            result['subtype']        = 'Rule Violation'
            return result

        # STAGE 2 — CNN-TCN Model
        if window_array is not None:
            pred = self.model.predict_window_with_subtype(
                window_array, flow_dict
            )

            # Fix — low confidence Flash Crowd may be Benign
            # Flash Crowd is derived from Benign — boundary is ambiguous
            if pred['class_id'] == 3 and pred['confidence'] < 0.75:
                pred['class_id'] = 0
                pred['label']    = 'Benign'
                pred['subtype']  = 'Normal Traffic'

            result['stages'].append({
                'stage':      'CNN-TCN Model',
                'decision':   pred['label'],
                'subtype':    pred['subtype'],
                'confidence': f"{pred['confidence']*100:.1f}%",
                'probs':      pred['probs']
            })

            result['subtype'] = pred['subtype']

            if pred['class_id'] != 0:
                self.stats['blocked_model'] += 1
                result['final_decision'] = 'BLOCK'
                result['final_reason']   = (
                    f"Model: {pred['label']} — "
                    f"{pred['subtype']} "
                    f"({pred['confidence']*100:.1f}%)"
                )
                self.rule_filter.record_outcome('Attack', rule_decision)
                return result

            # STAGE 3 — CUSUM Check
            rate   = flow_dict.get('Rate', 0)
            number = flow_dict.get('Number', 0)
            cusum_decision, cusum_info = self.cusum_filter.check(
                source_id, rate, number
            )
            result['stages'].append({
                'stage':    'CUSUM Check',
                'decision': cusum_decision,
                'info':     cusum_info
            })

            if cusum_decision == 'SUSPICIOUS':
                self.stats['suspicious'] += 1
                result['final_decision'] = 'SUSPICIOUS'
                result['final_reason']   = f'CUSUM drift: {cusum_info}'
                result['subtype']        = 'Mimicry Bot (CUSUM)'
                self.rule_filter.record_outcome('Attack', rule_decision)
                return result

            self.rule_filter.record_outcome('Benign', rule_decision)

        # All checks passed
        self.stats['allowed'] += 1
        result['final_decision'] = 'ALLOW'
        result['final_reason']   = 'Passed all checks'
        result['subtype']        = 'Normal Traffic'
        return result

    def get_stats(self):
        return self.stats

    def get_thresholds(self):
        return self.rule_filter.get_thresholds()