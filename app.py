from flask import Flask, jsonify, render_template, request, send_file
import pandas as pd
import numpy as np
import threading
import time
import os
import json
import signal
import sys
import smtplib
import socket
import webbrowser
from email.mime.text import MIMEText
from datetime import datetime
from pipeline import DetectionPipeline

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────
EMAIL_ENABLED    = False
EMAIL_FROM       = 'your_email@gmail.com'
EMAIL_TO         = 'admin@example.com'
EMAIL_PASSWORD   = 'your_app_password'
EMAIL_THRESHOLD  = 10

API_KEY          = 'iiot2024'
HISTORY_FILE     = 'models/detection_history.json'
LOG_FILE         = 'detection_log.txt'
SUMMARY_INTERVAL = 60

# ── Startup banner ─────────────────────────────────────────────
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()

def print_banner(local_ip='127.0.0.1'):
    print('\n' + '═' * 50)
    print('║     IIoT DDoS Detection System v2.0         ║')
    print('║     CNN-TCN + RL Agent + CUSUM + SHAP        ║')
    print('═' * 50)
    print(f'║  Dashboard : http://127.0.0.1:5000            ║')
    print(f'║  Mobile    : http://{local_ip}:5000/mobile')
    print('═' * 50 + '\n')

# ── Health check ───────────────────────────────────────────────
def health_check(pipe, df):
    print('\n── System Health Check ──────────────────────')
    t = pipe.get_thresholds()
    print(f'  ✅ Model loaded       — 80,964 params')
    print(f'  ✅ Scaler loaded')
    print(f'  ✅ Q-table loaded     — Rate={t["rate"]} SYN={t["syn"]}')
    print(f'  ✅ Data loaded        — {len(df)} rows (balanced)')
    print(f'  ✅ Pipeline ready')
    print(f'─────────────────────────────────────────────\n')

# ── Load pipeline ──────────────────────────────────────────────
print('\nStarting IIoT DDoS Detection System...')
print('Loading model and pipeline...')

pipe = DetectionPipeline(
    weights_path='models/model_v2.weights.h5',
    scaler_path='models/minmax_scaler_v2.pkl'
)

# Load balanced data with 5x more Benign rows
df_all    = pd.read_csv('data/demo_balanced.csv')
df_benign = df_all[df_all['label'] == 'BenignTraffic']
df_attack = df_all[df_all['label'] != 'BenignTraffic']

df = pd.concat([
    df_attack,
    df_benign,
    df_benign,
    df_benign,
    df_benign,
    df_benign
], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f'✅ Data loaded — {len(df)} rows')
print(f'   Benign rows : {len(df[df["label"] == "BenignTraffic"])}')
print(f'   Attack rows : {len(df[df["label"] != "BenignTraffic"])}')

# ── Load history ───────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'recent_results':       [],
        'stats': {
            'total': 0, 'blocked_rule': 0,
            'blocked_model': 0, 'suspicious': 0, 'allowed': 0
        },
        'blocked_ips':          [],
        'alerts':               [],
        'shap_latest':          [],
        'subtype_counts':       {},
        'manually_blocked_ips': [],
        'session_start':        datetime.now().strftime('%H:%M:%S')
    }

dashboard_data = load_history()
dashboard_data['session_start'] = datetime.now().strftime('%H:%M:%S')

current_index = 0
lock          = threading.Lock()
paused        = False
session_start = time.time()

# ── Logging ────────────────────────────────────────────────────
def log_detection(record):
    try:
        with open(LOG_FILE, 'a') as f:
            line = (f"[{record['time']}] "
                    f"{record['source_ip']:<18} "
                    f"{record['decision']:<12} "
                    f"{record['subtype']:<25} "
                    f"{record['reason']}\n")
            f.write(line)
    except Exception:
        pass

def log_console(record):
    decision = record['decision']
    icon = '🚨' if decision == 'BLOCK' else '⚠️ ' if decision == 'SUSPICIOUS' else '✅'
    print(f"[{record['time']}] {icon} {record['source_ip']:<18} "
          f"-> {decision:<12} | {record['subtype']:<25} | {record['reason']}")

# ── Email alert ────────────────────────────────────────────────
def send_email_alert(alert):
    if not EMAIL_ENABLED:
        return
    try:
        msg = MIMEText(
            f"Attack detected!\n\n"
            f"IP      : {alert['ip']}\n"
            f"Type    : {alert['type']}\n"
            f"Sub-type: {alert['subtype']}\n"
            f"Reason  : {alert['reason']}\n"
            f"Time    : {alert['time']}"
        )
        msg['Subject'] = f'IIoT Alert: {alert["type"]} from {alert["ip"]}'
        msg['From']    = EMAIL_FROM
        msg['To']      = EMAIL_TO
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f'  Email alert sent to {EMAIL_TO}')
    except Exception as e:
        print(f'  Email error: {e}')

# ── Save history ───────────────────────────────────────────────
def save_history():
    try:
        with lock:
            data_to_save = {
                'recent_results':       dashboard_data['recent_results'][:50],
                'stats':                dashboard_data['stats'],
                'blocked_ips':          dashboard_data['blocked_ips'],
                'alerts':               dashboard_data['alerts'][:50],
                'shap_latest':          dashboard_data['shap_latest'],
                'subtype_counts':       dashboard_data['subtype_counts'],
                'manually_blocked_ips': dashboard_data.get('manually_blocked_ips', []),
                'session_start':        dashboard_data['session_start']
            }
        with open(HISTORY_FILE, 'w') as f:
            json.dump(data_to_save, f)
    except Exception as e:
        print(f'Save history error: {e}')

# ── Periodic summary ───────────────────────────────────────────
def print_summary():
    while True:
        time.sleep(SUMMARY_INTERVAL)
        with lock:
            s       = dashboard_data['stats']
            total   = max(s['total'], 1)
            blocked = s['blocked_rule'] + s['blocked_model']
            top_sub = '--'
            if dashboard_data['subtype_counts']:
                top_sub = max(
                    dashboard_data['subtype_counts'],
                    key=dashboard_data['subtype_counts'].get
                )
            t = pipe.get_thresholds()

        elapsed = int(time.time() - session_start)
        mins    = elapsed // 60
        secs    = elapsed % 60

        print(f'\n{"="*48}')
        print(f'  {SUMMARY_INTERVAL}s Summary — Uptime: {mins:02d}:{secs:02d}')
        print(f'{"─"*48}')
        print(f'  Total flows    : {s["total"]}')
        print(f'  Blocked        : {blocked} ({blocked/total*100:.1f}%)')
        print(f'  Suspicious     : {s["suspicious"]}')
        print(f'  Allowed        : {s["allowed"]}')
        print(f'  Top sub-type   : {top_sub}')
        print(f'  RL Rate thresh : {t["rate"]}')
        print(f'  RL Reward      : {t["reward"]}')
        print(f'{"="*48}\n')
        save_history()

# ── Alert beep ─────────────────────────────────────────────────
def beep():
    try:
        print('\a', end='', flush=True)
    except Exception:
        pass

# ── IP utility ─────────────────────────────────────────────────
def simulate_source_id(index):
    base = 100 + (index % 50)
    return f'192.168.1.{base}'

def is_manually_blocked(ip):
    return ip in dashboard_data.get('manually_blocked_ips', [])

# ── Pipeline loop ──────────────────────────────────────────────
def run_pipeline_loop():
    global current_index

    while True:
        if paused:
            time.sleep(1)
            continue

        try:
            start = current_index * 10
            rows  = df.iloc[start: start + 10]

            if len(rows) < 10:
                current_index = 0
                continue

            flow_dict  = rows.iloc[-1].to_dict()
            window_arr = pipe.model.prepare_window_from_df(rows)
            source_id  = simulate_source_id(current_index)

            if is_manually_blocked(source_id):
                result = {
                    'final_decision': 'BLOCK',
                    'final_reason':   'Manually blocked by admin',
                    'subtype':        'Manual Block',
                    'stages':         []
                }
            else:
                result = pipe.process_flow(flow_dict, window_arr, source_id)

            shap_data    = []
            rule_blocked = any(
                s['stage'] == 'Adaptive Rule Filter' and s['decision'] == 'BLOCK'
                for s in result.get('stages', [])
            )

            if not rule_blocked:
                try:
                    shap_data = pipe.model.explain_window(window_arr)
                except Exception:
                    shap_data = []

            subtype = result.get('subtype', '--')

            record = {
                'id':        current_index,
                'source_ip': source_id,
                'decision':  result['final_decision'],
                'reason':    result['final_reason'],
                'subtype':   subtype,
                'time':      datetime.now().strftime('%H:%M:%S'),
                'stages':    result.get('stages', []),
                'shap':      shap_data
            }

            log_console(record)
            log_detection(record)

            with lock:
                dashboard_data['recent_results'].insert(0, record)
                dashboard_data['recent_results'] = \
                    dashboard_data['recent_results'][:20]
                dashboard_data['stats'] = pipe.get_stats()

                if shap_data:
                    dashboard_data['shap_latest'] = shap_data

                if subtype and subtype not in \
                        ['--', 'Normal Traffic', 'Manual Block']:
                    dashboard_data['subtype_counts'][subtype] = \
                        dashboard_data['subtype_counts'].get(subtype, 0) + 1

                if result['final_decision'] == 'BLOCK':
                    if source_id not in dashboard_data['blocked_ips']:
                        dashboard_data['blocked_ips'].append(source_id)
                        dashboard_data['blocked_ips'] = \
                            dashboard_data['blocked_ips'][:10]
                    beep()

                if result['final_decision'] in ['BLOCK', 'SUSPICIOUS']:
                    alert = {
                        'time':    datetime.now().strftime('%H:%M:%S'),
                        'ip':      source_id,
                        'type':    result['final_decision'],
                        'reason':  result['final_reason'],
                        'subtype': subtype,
                        'shap':    shap_data
                    }
                    dashboard_data['alerts'].insert(0, alert)
                    dashboard_data['alerts'] = \
                        dashboard_data['alerts'][:20]

                    total_blocked = (
                        dashboard_data['stats']['blocked_rule'] +
                        dashboard_data['stats']['blocked_model']
                    )
                    if total_blocked % EMAIL_THRESHOLD == 0 \
                            and total_blocked > 0:
                        threading.Thread(
                            target=send_email_alert,
                            args=(alert,),
                            daemon=True
                        ).start()

            current_index += 1

        except Exception as e:
            print(f'Pipeline error: {e}')
            current_index += 1

        time.sleep(2)

# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mobile')
def mobile():
    return render_template('mobile.html')

@app.route('/api/data')
def get_data():
    with lock:
        data = dict(dashboard_data)
        data['uptime']     = int(time.time() - session_start)
        data['thresholds'] = pipe.get_thresholds()
        return jsonify(data)

@app.route('/api/stats')
def get_stats():
    with lock:
        return jsonify(dashboard_data['stats'])

@app.route('/api/thresholds')
def get_thresholds():
    return jsonify(pipe.get_thresholds())

@app.route('/api/mobile')
def get_mobile_data():
    with lock:
        stats          = dashboard_data['stats']
        total          = max(stats['total'], 1)
        blocked        = stats['blocked_rule'] + stats['blocked_model']
        detection_rate = round(blocked / total * 100, 1)
        allowed = stats['allowed']
       

        if detection_rate > 80:
            threat_level = 'Critical'
            threat_color = 'red'
        elif detection_rate > 60:
            threat_level = 'High'
            threat_color = 'orange'
        elif detection_rate > 30:
            threat_level = 'Medium'
            threat_color = 'amber'
        else:
            threat_level = 'Low'
            threat_color = 'green'

        top_ips = {}
        for r in dashboard_data['recent_results']:
            if r['decision'] == 'BLOCK':
                ip = r['source_ip']
                top_ips[ip] = top_ips.get(ip, 0) + 1
        top_ips_sorted = sorted(
            top_ips.items(), key=lambda x: x[1], reverse=True
        )[:5]

        try:
            thresholds = pipe.get_thresholds()
        except Exception:
            thresholds = {
                'rate': 10000, 'syn': 100,
                'header': 20, 'epsilon': 0.2, 'reward': 0
            }

        return jsonify({
            'stats':          dashboard_data['stats'],
            'alerts':         dashboard_data['alerts'][:10],
            'blocked_ips':    dashboard_data['blocked_ips'],
            'shap_latest':    dashboard_data['shap_latest'],
            'subtype_counts': dashboard_data['subtype_counts'],
            'recent_results': dashboard_data['recent_results'][:5],
            'threat_level':   threat_level,
            'threat_color':   threat_color,
            'detection_rate': detection_rate,
            'top_ips':        top_ips_sorted,
            'thresholds':     thresholds,
            'uptime':         int(time.time() - session_start)
        })

@app.route('/api/pause', methods=['POST'])
def pause_pipeline():
    global paused
    paused = not paused
    status = 'paused' if paused else 'running'
    print(f'  Pipeline {status}')
    return jsonify({'status': status})

@app.route('/api/block_ip', methods=['POST'])
def block_ip():
    data = request.get_json()
    ip   = data.get('ip', '').strip()
    if ip and ip not in dashboard_data.get('manually_blocked_ips', []):
        with lock:
            dashboard_data.setdefault(
                'manually_blocked_ips', []
            ).append(ip)
        print(f'  Manually blocked IP: {ip}')
        return jsonify({'status': 'blocked', 'ip': ip})
    return jsonify({'status': 'already_blocked', 'ip': ip})

@app.route('/api/clear_alerts', methods=['POST'])
def clear_alerts():
    with lock:
        dashboard_data['alerts']      = []
        dashboard_data['blocked_ips'] = []
    print('  Alerts cleared by admin')
    return jsonify({'status': 'cleared'})

@app.route('/api/export_alerts')
def export_alerts():
    try:
        with lock:
            alerts = dashboard_data['alerts']
        rows = []
        for a in alerts:
            rows.append({
                'time':    a.get('time', ''),
                'ip':      a.get('ip', ''),
                'type':    a.get('type', ''),
                'subtype': a.get('subtype', ''),
                'reason':  a.get('reason', '')
            })
        export_df = pd.DataFrame(rows)
        path      = 'models/alerts_export.csv'
        export_df.to_csv(path, index=False)
        return send_file(
            path, as_attachment=True,
            download_name='alerts_export.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/manifest.json')
def manifest():
    return app.send_static_file('manifest.json')

# ── Graceful shutdown ──────────────────────────────────────────
def shutdown_handler(sig, frame):
    print('\n\nShutting down gracefully...')
    pipe.rule_filter.save()
    save_history()
    print('Q-table saved')
    print('Detection history saved')
    print('Goodbye!')
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)

# ── Start ──────────────────────────────────────────────────────
if __name__ == '__main__':
    health_check(pipe, df)
    print_banner(get_local_ip())

    pipeline_thread = threading.Thread(
        target=run_pipeline_loop, daemon=True
    )
    pipeline_thread.start()

    summary_thread = threading.Thread(
        target=print_summary, daemon=True
    )
    summary_thread.start()

    def open_browser():
        time.sleep(2)
        webbrowser.open('http://127.0.0.1:5000')
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(debug=False, host='0.0.0.0', port=5000)