FROM ubuntu:latest

RUN apt-get update \
  && apt-get -y install \
    cron \
    python3 \
    python3-pip

WORKDIR /pull_policies

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade --no-cache-dir -r requirements.txt

COPY . .
 
# execution permissions
RUN chmod 0644 ./launch-cron
# must execute for each script
RUN chmod 0744 ./src/pull_policies.py

# apply cron job
#RUN crontab ./launch-cron
 
# cron log
RUN touch /var/log/cron.log
 
# tail cron log
#CMD cron && tail -f /var/log/cron.log
CMD python3 ./src/pull_policies.py