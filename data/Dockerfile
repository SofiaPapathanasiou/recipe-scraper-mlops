FROM python:3.10-slim
WORKDIR /app
RUN pip install python-swiftclient python-keystoneclient psycopg2-binary pyyaml pandas numpy requests faker tqdm
COPY scripts/ /app/scripts/
COPY quality/ /app/quality/
CMD ["bash"]
