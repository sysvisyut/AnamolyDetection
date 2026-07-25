<<<<<<< HEAD
# AI-Powered Behavioral Anomaly Detection for Cybersecurity

This is the repository for the AI-Powered Behavioral Anomaly Detection hackathon project.

## Project Description

A machine learning pipeline for detecting cybersecurity behavioral anomalies such as insider threats, lateral movement, and impossible travel. Designed for integration with an analyst dashboard to surface explainable anomaly alerts in real-time.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd anomaly_detection
    ```

2.  **Set up the environment:**
    Create a `.env` file based on the example:
    ```bash
    cp .env.example .env
    ```
    Ensure `SECRET_KEY` is populated for production.

3.  **Install dependencies:**
    We recommend using a virtual environment (Python 3.11+ required).
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -e ".[dev]"
    ```
    *Note: `pip-tools` can be used to generate `requirements.txt` from `pyproject.toml`.*

4.  **Run Tests:**
    ```bash
    pytest
    ```

5.  **Run the application (Once implemented):**
    ```bash
    uvicorn anomaly_detection.api.main:app --reload
    ```
=======
# AnamolyDetection
>>>>>>> aec4b9b3794289a951a2165187cd880b25364074
