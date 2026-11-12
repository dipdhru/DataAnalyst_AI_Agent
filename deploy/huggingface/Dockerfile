# Hugging Face Space (Docker SDK) for the public demo. Spaces run the container as user 1000.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    ANALYST_DEMO=1 ANALYST_CONFIG=/home/user/app/config.yaml
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user src ./src
COPY --chown=user ui ./ui
COPY --chown=user config.yaml metrics.yaml ./
RUN uv sync --frozen --no-dev

EXPOSE 7860
CMD ["uv", "run", "--no-dev", "streamlit", "run", "ui/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
