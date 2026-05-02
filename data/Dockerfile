FROM python:3.10-slim
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install python-swiftclient python-keystoneclient psycopg2-binary pyyaml pandas numpy requests faker tqdm
COPY scripts/ /app/scripts/
COPY quality/ /app/quality/
CMD ["bash"]
