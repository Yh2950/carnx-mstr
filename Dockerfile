# CARN-X  --  container for Hugging Face Spaces (sdk: docker) and any VPS.
# Pins Python 3.11 + the exact scientific stack the code is written against, so
# the deploy behaves like the local run -- no "works here, breaks there".

FROM python:3.11-slim

# non-root user (Hugging Face convention; keeps site-packages writable for the
# index.html icon/LTR patches in brand_boot.py / ltr_boot.py)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    CARNX_ON_CLOUD=1

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read()==b'ok' else 1)"
CMD ["streamlit", "run", "mstr_app.py"]
