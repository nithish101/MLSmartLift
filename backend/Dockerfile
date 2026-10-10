# syntax=docker/dockerfile:1

# ---- build: resolve and install dependencies + the app into a venv ----------
FROM python:3.12-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /src
# Dependencies first so they cache independently of app code.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY app ./app
# uv keys its cached wheel of a local project on pyproject.toml only, so force a
# rebuild of the app itself or a warm cache would ship stale source.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --reinstall-package smartlift --refresh-package smartlift

# ---- runtime: just the interpreter and the venv, as an unprivileged user ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --system --gid 10001 smartlift \
    && useradd --system --uid 10001 --gid smartlift --home-dir /app --create-home smartlift

COPY --from=build --chown=smartlift:smartlift /opt/venv /opt/venv

WORKDIR /app
USER smartlift

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=10 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
