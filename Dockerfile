FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    BANTAY_TRUSTED_HOSTS=* \
    BANTAY_TRUSTED_PROXY=1 \
    BANTAY_DB_PATH=/var/lib/bantay-kontrata/reviews.sqlite3

WORKDIR /app

COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /var/lib/bantay-kontrata \
    && chown -R appuser:appuser /var/lib/bantay-kontrata /app

COPY --chown=appuser:appuser . .

USER appuser

VOLUME ["/var/lib/bantay-kontrata"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health')"

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 wsgi:app"]
