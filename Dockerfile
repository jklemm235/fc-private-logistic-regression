FROM python:3.8-buster

RUN apt-get update && apt-get install -y supervisor nginx

RUN pip3 install --upgrade pip

COPY server_config/supervisord.conf /supervisord.conf
COPY server_config/nginx /etc/nginx/sites-available/default
COPY server_config/docker-entrypoint.sh /entrypoint.sh

COPY requirements.txt /app/requirements.txt
COPY FeatureCloud-0.0.18.tar.gz /app/FeatureCloud-0.0.18.tar.gz
RUN python3 -m pip install ./app/FeatureCloud-0.0.18.tar.gz
RUN pip3 install -r ./app/requirements.txt

COPY . /app

EXPOSE 9000 9001

ENTRYPOINT ["sh", "/entrypoint.sh"]
