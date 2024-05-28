FROM python:3.9
WORKDIR /app
COPY . .

ENV PROJECT_ID=acn-amex-account-poc-sandbox
ENV LOCATION=us-central1
RUN pip install gunicorn
RUN pip install -r requirements.txt
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app