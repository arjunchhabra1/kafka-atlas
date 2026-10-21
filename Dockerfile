FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Dependencies first (cached unless the lockfile changes), then the project itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY migrations ./migrations
COPY config.yaml ./
RUN uv sync --frozen --no-dev

# No secrets or data in the image: configuration comes from env vars / Secret Manager at runtime.
RUN useradd --uid 10001 --no-create-home kgrag
USER kgrag

ENV PORT=8080
EXPOSE 8080
CMD ["kgrag", "serve"]
