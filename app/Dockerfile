FROM python:3.8-slim-buster as app

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=src/app.py

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]