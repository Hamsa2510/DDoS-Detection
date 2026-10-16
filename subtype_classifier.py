# ================================================================
# subtype_classifier.py
# Rule-based sub-type classifier
# Runs AFTER CNN-TCN model identifies the broad class
# Identifies the specific attack variant using feature thresholds
# ================================================================

class SubtypeClassifier:

    def classify(self, flow_dict, broad_class):
        """
        flow_dict   : dict of raw feature values from the last row of window
        broad_class : string — 'High-Rate DDoS', 'Low-Rate DDoS',
                                'Benign', 'Flash Crowd'
        Returns: subtype string
        """
        if broad_class == 'Benign':
            return 'Normal Traffic'

        if broad_class == 'Flash Crowd':
            return 'Traffic Surge'

        if broad_class == 'High-Rate DDoS':
            return self._classify_high_rate(flow_dict)

        if broad_class == 'Low-Rate DDoS':
            return self._classify_low_rate(flow_dict)

        return 'Unknown'

    def _classify_high_rate(self, f):
        """Identify specific High-Rate DDoS sub-type."""

        syn  = f.get('syn_flag_number', 0)
        rst  = f.get('rst_flag_number', 0)
        fin  = f.get('fin_flag_number', 0)
        psh  = f.get('psh_flag_number', 0)
        ack  = f.get('ack_flag_number', 0)
        udp  = f.get('UDP', 0)
        icmp = f.get('ICMP', 0)
        tcp  = f.get('TCP', 0)
        ipv  = f.get('IPv', 0)
        rate = f.get('Rate', 0)
        hlen = f.get('Header_Length', 64)
        iat  = f.get('IAT', 1)

        # Fragmentation attacks — tiny header length
        if hlen < 30:
            if icmp > 0:
                return 'ICMP Fragmentation'
            if udp > 0:
                return 'UDP Fragmentation'
            if ack > 0:
                return 'ACK Fragmentation'

        # Mirai variants — very high rate + specific protocols
        if rate > 5000 and icmp > 0 and ipv > 0:
            return 'Mirai GRE-IP Flood'
        if rate > 5000 and icmp > 0:
            return 'Mirai GRE-ETH Flood'
        if rate > 3000 and udp > 0 and iat < 0.01:
            return 'Mirai UDP Plain'

        # SynonymousIP — high syn + high rate + high ipv
        if syn > 50 and rate > 3000 and ipv > 0:
            return 'SynonymousIP Flood'

        # Standard flood types
        if syn > 50:
            return 'SYN Flood'
        if rst > 30 and fin > 30:
            return 'RST-FIN Flood'
        if psh > 30 and ack > 30:
            return 'PSH-ACK Flood'
        if icmp > 0 and rate > 1000:
            return 'ICMP Flood'
        if udp > 0 and rate > 1000:
            return 'UDP Flood'
        if tcp > 0 and rate > 1000:
            return 'TCP Flood'

        # Fallback
        return 'High-Rate Flood'

    def _classify_low_rate(self, f):
        """Identify specific Low-Rate DDoS sub-type."""

        rate = f.get('Rate', 0)
        http = f.get('HTTP', 0)
        dns  = f.get('DNS', 0)
        syn  = f.get('syn_flag_number', 0)
        udp  = f.get('UDP', 0)
        tcp  = f.get('TCP', 0)
        iat  = f.get('IAT', 1)

        # HTTP based
        if http > 0 and rate < 500:
            return 'HTTP Flood (Slowloris)'

        # DNS flood
        if dns > 0 and rate < 500:
            return 'DNS Flood'

        # Low rate SYN
        if syn > 5 and rate < 200:
            return 'DoS SYN Flood'

        # Low rate UDP
        if udp > 0 and rate < 300:
            return 'DoS UDP Flood'

        # Low rate TCP
        if tcp > 0 and rate < 300:
            return 'DoS TCP Flood'

        # Very slow — likely Slowloris
        if iat > 1.0 and rate < 100:
            return 'Slowloris'

        return 'Low-Rate Flood'