# Monitoring dan Logging — Yusfitasari

Panduan lengkap setup Prometheus + Grafana untuk monitoring sistem ML Heart Disease.

## 📁 Struktur Folder Submission

```
Monitoring dan Logging/
├── 1.bukti_serving/                ← Screenshot serving (isi sendiri)
├── 2.prometheus.yml                ← Konfigurasi Prometheus
├── 3.prometheus_exporter.py        ← Custom exporter metrik sistem
├── 4.bukti monitoring Prometheus/  ← Screenshot Prometheus (isi sendiri)
├── 5.bukti monitoring Grafana/     ← Screenshot Grafana (isi sendiri)
├── 6.bukti alerting Grafana/       ← Screenshot alerting (isi sendiri)
└── 7.inference.py                  ← Flask serving + Prometheus metrics
```

---

## 🚀 Cara Setup & Jalankan

### 1. Install Dependencies
```bash
pip install flask mlflow dagshub prometheus-client psutil
```

### 2. Jalankan Inference Server (Serving Model)
```bash
python 7.inference.py
# Berjalan di: http://localhost:5001
# Metrics di : http://localhost:5001/metrics
```

### 3. Jalankan Prometheus Exporter (Metrik Sistem)
```bash
python 3.prometheus_exporter.py
# Berjalan di: http://localhost:8000/metrics
```

### 4. Download & Jalankan Prometheus
```bash
# Download dari https://prometheus.io/download/
# Salin prometheus.yml ke folder prometheus
./prometheus --config.file=2.prometheus.yml
# UI di: http://localhost:9090
```

### 5. Download & Jalankan Grafana
```bash
# Download dari https://grafana.com/grafana/download
# Default: http://localhost:3000 (admin/admin)
```

---

## 📊 Metrik yang Tersedia (12 metrik total → Advance ≥ 10)

| # | Nama Metrik | Tipe | Deskripsi |
|---|---|---|---|
| 1 | `heart_disease_request_total` | Counter | Total request masuk |
| 2 | `heart_disease_request_latency_seconds` | Histogram | Latensi request |
| 3 | `heart_disease_prediction_total` | Counter | Prediksi per kelas |
| 4 | `heart_disease_error_total` | Counter | Total error |
| 5 | `heart_disease_prediction_probability` | Gauge | Probabilitas prediksi terakhir |
| 6 | `heart_disease_model_loaded` | Gauge | Status model |
| 7 | `heart_disease_model_load_seconds` | Gauge | Waktu load model |
| 8 | `heart_disease_requests_in_progress` | Gauge | Request sedang diproses |
| 9 | `heart_disease_request_size_bytes` | Summary | Ukuran payload |
| 10 | `heart_disease_input_feature_count` | Histogram | Jumlah fitur input |
| 11 | `heart_disease_positive_prediction_total` | Counter | Total prediksi positif |
| 12 | `heart_disease_negative_prediction_total` | Counter | Total prediksi negatif |

---

## 🔔 Alerting Grafana (3 alert untuk Advance)

Buat 3 alert berikut di Grafana:

### Alert 1 — High Error Rate
- **Query:** `rate(heart_disease_error_total[5m]) > 0.1`
- **Kondisi:** Lebih dari 0.1 error/detik dalam 5 menit terakhir

### Alert 2 — High Latency
- **Query:** `histogram_quantile(0.95, heart_disease_request_latency_seconds_bucket) > 2`
- **Kondisi:** P95 latency > 2 detik

### Alert 3 — High Positive Prediction Rate
- **Query:** `rate(heart_disease_positive_prediction_total[10m]) / rate(heart_disease_request_total[10m]) > 0.8`
- **Kondisi:** Lebih dari 80% prediksi bernilai positif dalam 10 menit

---

## 📸 Screenshot yang Harus Dikumpulkan

| Folder | Isi |
|---|---|
| `1.bukti_serving/` | Screenshot terminal/browser saat model serving aktif |
| `4.bukti monitoring Prometheus/` | Screenshot ≥ 10 metrik di Prometheus UI |
| `5.bukti monitoring Grafana/` | Screenshot dashboard Grafana dengan ≥ 10 panel |
| `6.bukti alerting Grafana/` | Screenshot rules + notifikasi untuk 3 alert |

> ⚠️ **Penting:** Pastikan nama username Dicoding kamu terlihat di dashboard Grafana!

---

## 🧪 Test Prediksi

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    "cp_0": 0, "cp_1": 0, "cp_2": 0, "cp_3": 1,
    "restecg_0": 0, "restecg_1": 1, "restecg_2": 0,
    "slope_0": 0, "slope_1": 1, "slope_2": 0,
    "ca_0": 1, "ca_1": 0, "ca_2": 0, "ca_3": 0,
    "thal_0": 0, "thal_1": 0, "thal_2": 1, "thal_3": 0
  }'
```
