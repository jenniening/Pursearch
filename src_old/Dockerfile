FROM python:3
WORKDIR /app
COPY . /app
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y git python-pip
RUN pip install --upgrade pip
RUN apt-get install -y protobuf-compiler python-pil python-lxml
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models
WORKDIR /tensorflow/models/research/
RUN protoc object_detection/protos/*.proto --python_out=.
RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
WORKDIR /app
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["app.py"]

