FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
#RUN pip install --no-cache-dir --default-timeout=200 -r requirements.txt
#RUN pip install --no-cache-dir --default-timeout=200 -r requirements.txt
#RUN pip install --no-cache-dir scikit-learn && \
#    pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    -i https://pypi.org/simple \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple \
    -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8032"]