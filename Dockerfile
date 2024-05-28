FROM python:3.9
RUN pip install gunicorn
RUN pip install -r requirements.txt
ENV PROJECT_ID=acn-amex-account-poc-sandbox
ENV LOCATION=us-central1
ENV PORT=8080
WORKDIR /app
COPY . .
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app