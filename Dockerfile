FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY chatbot/ ./chatbot/
COPY data/pdf/ ./data/pdf/

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "chatbot/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]