# syntax=docker/dockerfile:1

# faithgate is a batch CLI: score a JSONL file, calibrate, write a report. The image ships the CLI
# and the bundled examples; the NLI model and RAGTruth are fetched into /data at run time.

# Stage 1 pins the runtime dependencies from uv.lock, installs them into /venv with pip, then
# builds faithgate as a wheel and adds it. uv stays in this stage.
FROM python:3.12-slim AS wheel
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --output-file requirements.txt \
    && python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --requirement requirements.txt
COPY README.md LICENSE ./
COPY src ./src
RUN uv build --wheel --out-dir dist \
    && /venv/bin/pip install --no-cache-dir --no-deps dist/*.whl

# Stage 2 copies the venv onto a fresh slim base. Downloads go to /data, the user's home.
FROM python:3.12-slim
RUN groupadd --gid 10001 gate \
    && useradd --uid 10001 --gid gate --home-dir /data --no-create-home \
       --shell /usr/sbin/nologin gate \
    && install --directory --owner=gate --group=gate /data /work
COPY --from=wheel /venv /venv
COPY --chown=gate:gate examples /work/examples
ENV PATH=/venv/bin:$PATH \
    FAITHGATE_DATA_DIR=/data \
    HF_HOME=/data/hf \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
USER 10001:10001
VOLUME ["/data"]
# No HEALTHCHECK: each container runs one command and exits, so there is nothing to probe.
ENTRYPOINT ["faithgate"]
CMD ["--help"]
