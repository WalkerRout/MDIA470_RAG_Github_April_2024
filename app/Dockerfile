FROM python:3.10-slim-bullseye as app

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=src/app.py

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]