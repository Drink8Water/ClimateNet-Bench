FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app:/app/src

COPY requirements-api.txt requirements.txt pyproject.toml ./
COPY alembic.ini ./
COPY alembic ./alembic
COPY backend ./backend
COPY src ./src

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-api.txt

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
