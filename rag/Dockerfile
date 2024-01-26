FROM ubuntu:20.04 as lib-rag

RUN apt-get update \
  && apt-get install -y \
    python3 \
    python3-pip

ENV FIREWORKS_API_KEY ${FIREWORKS_API_KEY}

EXPOSE 8333

WORKDIR workspace/

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "app.py"]