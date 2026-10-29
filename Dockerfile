FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml /app/
COPY tohe /app/tohe
RUN pip install --no-cache-dir .
COPY configs /app/configs
COPY scripts /app/scripts
COPY tests /app/tests
CMD ["python", "scripts/run_experiment.py", "--help"]
