FROM nvcr.io/nvidia/tensorrt:23.06-py3

RUN mkdir -p /app && mkdir -p /app/models
WORKDIR /app

ADD VERSION /app/
ADD app.py /app/
ADD roop2.py /app/
ADD requirements_cuda.txt /app/

RUN pip install -r /app/requirements_cuda.txt

CMD [ "executable" ]