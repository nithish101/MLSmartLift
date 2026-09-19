# Locust load generator + performance gate. Built from the backend context so the
# locustfile can sample realistic feature vectors from app.ml.generate_data.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group loadtest --no-install-project
COPY app ./app
COPY loadtest ./loadtest
RUN uv sync --frozen --no-dev --group loadtest
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["/app/loadtest/entrypoint.sh"]
