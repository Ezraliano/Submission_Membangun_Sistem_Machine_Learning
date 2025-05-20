from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# ---------- METRIK UTAMA ----------
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency (seconds)')
THROUGHPUT = Counter('http_throughput_total', 'HTTP Request Throughput (requests per second)')

# ---------- METRIK SISTEM ----------
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage_percent', 'RAM Usage Percentage')

# ---------- ENDPOINT MONITORING ----------
@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))  # CPU
    RAM_USAGE.set(psutil.virtual_memory().percent)  # RAM
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# ---------- ENDPOINT PREDIKSI ----------
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    # Simulasi panggil API model kamu (pastikan endpoint aktif di MLflow atau FastAPI)
    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- MAIN ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
