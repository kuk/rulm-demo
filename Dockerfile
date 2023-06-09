FROM python:3.10.8-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY instruct.py .
ENTRYPOINT ["python", "instruct.py"]
