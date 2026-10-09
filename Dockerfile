# syntax=docker/dockerfile:1

# Build stage: resolve the locked dependencies into a virtual environment.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Runtime stage: the same slim base, no build tools, unprivileged user.
FROM python:3.12-slim
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin faithgate \
    && mkdir /data \
    && chown 10001:10001 /data
COPY --from=build /opt/venv /opt/venv
COPY --chown=10001:10001 examples /app/examples
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    FAITHGATE_DATA_DIR=/data \
    HF_HOME=/data/hf
WORKDIR /app
USER 10001
VOLUME ["/data"]
# A batch CLI, not a long-running service, so there is no HEALTHCHECK to run.
ENTRYPOINT ["faithgate"]
CMD ["--help"]
