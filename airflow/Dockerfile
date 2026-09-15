# AIR-001 - container standalone, separado do docker-compose.yml do OBS-001 (menor blast
# radius: não mexe na stack já validada). Airflow 2.x (TaskFlow API), não 3.x - ver
# docs/specs/AIR-001.md, "Decisão: ambiente".
FROM apache/airflow:2.10.5-python3.12

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
