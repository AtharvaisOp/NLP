"""Run the demonstrable API flow without assuming smoke model accuracy.

Usage: python scripts/smoke-api.py https://mahapulse-staging-api.onrender.com
This inserts a small synthetic batch into the named staging database.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('base_url', nargs='?', default='http://localhost:8000')
args = parser.parse_args()
base = args.base_url.rstrip('/')

def call(path, *, body=None, content_type=None, expected=200):
    request = Request(base + path, data=body, headers={'Content-Type': content_type} if content_type else {})
    started = perf_counter()
    try:
        response = urlopen(request, timeout=300)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read()
        assert response.status == expected, f'{path}: HTTP {response.status}, expected {expected}'
        elapsed = round((perf_counter() - started) * 1000, 2)
        return raw, dict(response.headers), elapsed

def payload(path, **kwargs):
    raw, _, elapsed = call(path, **kwargs)
    return json.loads(raw), elapsed

report = {'operations': {}, 'single': []}
for endpoint in ['/health', '/ready', '/v1/model-info']:
    data, elapsed = payload(endpoint)
    report['operations'][endpoint] = {'latency_ms': elapsed, 'data': data}
texts = ['हा मोबाईल खूप चांगला आहे.', 'ही सेवा अत्यंत खराब आहे.', 'आज दुकान सकाळी दहा वाजता उघडले.', 'हा phone चांगला आहे पण battery backup खराब आहे.']
for text in texts + ['हा अनुभव चांगला आहे. ' * 100]:
    data, elapsed = payload('/v1/analyze', body=json.dumps({'text': text}, ensure_ascii=False).encode('utf-8'), content_type='application/json')
    assert data['sentiment']['label'] in {'positive', 'negative', 'neutral'}
    assert abs(sum(data['sentiment']['probabilities'].values()) - 1) < .001
    report['single'].append({'sentiment': data['sentiment'], 'model_version': data['meta']['model_version'], 'latency_ms': elapsed, 'keyword_count': len(data['keywords']), 'summary_provider': data['summary']['provider'], 'warnings': data['meta']['warnings']})
for text in ['', '   ']:
    call('/v1/analyze', body=json.dumps({'text': text}).encode(), content_type='application/json', expected=422)
buffer = io.StringIO(newline='')
writer = csv.writer(buffer)
writer.writerow(['text'])
writer.writerows([[text] for text in texts + ['']])
boundary = 'mahapulse-' + uuid4().hex
body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="marathi-smoke.csv"\r\nContent-Type: text/csv\r\n\r\n'.encode() + buffer.getvalue().encode('utf-8') + f'\r\n--{boundary}--\r\n'.encode())
batch, elapsed = payload('/v1/analyze/batch?text_column=text', body=body, content_type=f'multipart/form-data; boundary={boundary}')
assert batch['status'] == 'partial' and batch['successful_documents'] == 4 and batch['failed_documents'] == 1, batch
sid = batch['session_id']
report['batch'] = {'result': batch, 'latency_ms': elapsed}
rows = []
for offset in [0, 2, 4]:
    page, _ = payload(f'/v1/analyses/{sid}?limit=2&offset={offset}')
    assert page['offset'] == offset and page['limit'] == 2
    assert len(page['documents']) <= 2
    rows.extend(page['documents'])
assert len(rows) == 5 and len({row['id'] for row in rows}) == 5
report['pagination'] = {'rows': len(rows), 'privacy_omits_original': all(row['original_text'] is None for row in rows)}
analytics, _ = payload(f'/v1/analyses/{sid}/analytics')
assert analytics['total'] == 5 and analytics['successful'] == 4 and analytics['failed'] == 1
assert sum(value['count'] for value in analytics['sentiment'].values()) == 4
assert analytics['summary'] is None
report['analytics'] = analytics
json_export, _, _ = call(f'/v1/analyses/{sid}/export?format=json')
assert len(json.loads(json_export)['documents']) == 5
csv_export, headers, _ = call(f'/v1/analyses/{sid}/export?format=csv')
assert 'csv' in {key.lower(): value for key, value in headers.items()}.get('content-type', '')
assert len(list(csv.DictReader(io.StringIO(csv_export.decode('utf-8-sig'))))) == 5
report['exports'] = {'json_documents': 5, 'csv_rows': 5, 'csv_bytes': len(csv_export), 'json_bytes': len(json_export)}
print(json.dumps(report, ensure_ascii=False, indent=2))
