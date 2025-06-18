# 🧭 Data Anomaly Hunter

**Real-Time Anomaly Detection Dashboard** built for monitoring system metrics like CPU, memory, I/O, and network traffic using Machine Learning + Rule Engines + Real-time UI — with webhook alerts and export capability.

> 🚀 Built with: **Python, Streamlit, Scikit-learn, Prophet, YAML Rule Engine, Zapier Webhooks**

---

## 📌 Features

### ✅ Anomaly Detection Methods
- **Isolation Forest** (scikit-learn): Tree-based outlier detection using multi-metric input.
- **Prophet (Facebook)**: Forecasting + anomaly detection on time series (CPU utilization).
- **Rule Engine**: Editable no-code logic (YAML rules with real-time sliders in sidebar).

### ✅ Interactive Dashboard
- **Live visualizations** of metrics & Prophet predictions.
- **Anomaly tables** with color-coded flags.
- **Editable Rule Editor UI**: change thresholds dynamically via sidebar.
- **Model selection**: choose Isolation Forest, Prophet, or Both.

### ✅ Alerting System (Zapier Webhook)
- On detecting an anomaly, the system **POSTs to a Zapier webhook**.
- Zapier integrates the alert with **Google Sheets** or any other downstream action.
- Success message shown in UI: `🚀 Webhook Triggered!`

### ✅ Export Functionality
- Export detected anomalies as `.csv` or `.json`.
- Option to export all results or **only anomalies**.

---

![image](https://github.com/user-attachments/assets/06470687-7a1d-44aa-80d6-6a0df1c476d9)

## 🧠 Tech Stack

| Layer        | Tools Used                                  |
|--------------|----------------------------------------------|
| Backend      | Python, Pandas                              |
| Models       | Isolation Forest, Prophet                   |
| Rules Engine | YAML config + custom parser                 |
| Dashboard    | Streamlit                                   |
| Alerting     | Zapier Webhooks (connected to Google Sheets)|
| Export       | Streamlit’s file download API               |

---

## 📊 Sample Use Case

**Problem:** Enterprise systems generate high-volume metrics (CPU, memory, disk I/O). Manual anomaly detection is slow and error-prone.

**Solution:** This app:
- Detects anomalies in real-time
- Lets ops teams configure logic without code
- Sends critical anomalies to Google Sheets (via Zapier) for further action or audit

---

## 🛠 Setup & Run

### Clone the Repo, install dependencies and run the app

```bash
git clone https://github.com/your_username/data-anomaly-hunter.git
cd data-anomaly-hunter

pip install -r requirements.txt

streamlit run app.py
```

### 🔗 Zapier Integration Guide
Create a Zapier account.

Make a new Zap:
Trigger: Webhook → Catch Hook
Action: Google Sheets → Create Row
Paste the webhook URL in app.py inside the send_to_zapier_webhook() function.
Run the app and trigger anomalies to test.
✅ Check utils/webhook.py to customize payloads.

### File Structure
```bash
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── data/
│   └── stream.csv            # Input data
├── rules/
│   └── rules.yaml            # Editable rule logic
├── utils/
│   ├── anomaly_detector.py   # Isolation Forest logic
│   ├── prophet_detector.py   # Prophet-based detector
│   ├── rule_engine.py        # Rule engine (YAML based)
│   └── webhook.py            # Zapier webhook function
```

### Future Enhancements
Kafka-based live stream ingestion
Alert throttling and deduplication
Slack/Email integration for alerting
Auto-retraining model pipeline
Historical anomaly analytics dashboard

### 🙌 Author
Sarvesh,
An engineer on a mission to build smart, production-grade ML systems 🚀
