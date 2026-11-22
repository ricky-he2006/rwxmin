# Same base image the commits were labeled in (its Python decides which tests run).
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# The exact package set the labels were produced with (PyPI only).
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

# sqlglot at the pinned commit, from its own GitHub repository.
ARG SQLGLOT_SHA=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
RUN python - <<PY
import io, tarfile, urllib.request, shutil, os
url = "https://github.com/tobymao/sqlglot/archive/${SQLGLOT_SHA}.tar.gz"
data = urllib.request.urlopen(url, timeout=120).read()
with tarfile.open(fileobj=io.BytesIO(data)) as tf:
    tf.extractall("/opt", filter="data")
shutil.move("/opt/sqlglot-${SQLGLOT_SHA}", "/opt/sqlglot")
PY
# Precompile so imports at run time (read-only disk) are fast.
RUN python -m compileall -q /opt/sqlglot/sqlglot /opt/sqlglot/tests && chmod -R a+rX /opt/sqlglot

# baseline memo of the optimizer pipeline (its own layer: it only depends on these files)
COPY app/canon.py app/optmemo.py app/buildopt.py /app/
RUN BR_REPO=/opt/sqlglot PYTHONDONTWRITEBYTECODE=1 python /app/buildopt.py /app/optmemo \
    || (rm -f /app/optmemo.idx /app/optmemo.bin; echo "optimizer memo build failed: running without it")
COPY app /app
# baseline memo for the TPC executor tests, computed from the pinned checkout
RUN BR_REPO=/opt/sqlglot PYTHONDONTWRITEBYTECODE=1 python /app/buildmemo.py /app/memo.pkl \
    || (rm -f /app/memo.pkl; echo "memo build failed: running without it")
RUN BR_REPO=/opt/sqlglot PYTHONDONTWRITEBYTECODE=1 python /app/buildqueryattrs.py /app/queryattrs.pkl
RUN BR_REPO=/opt/sqlglot PYTHONDONTWRITEBYTECODE=1 python /app/buildvalidators.py /app/validatorcalls.idx
RUN BR_DATA=/nonexistent python /app/prep.py || echo "map preparation failed: maps are parsed at run time"
RUN python -m compileall -q /app && chmod -R a+rX /app
ENV SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1
CMD ["python", "/app/selector.py"]
