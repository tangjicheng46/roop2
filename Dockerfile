FROM python:3.10

RUN mkdir -p /app
WORKDIR /app 

ADD VERSION /app/VERSION
ADD app.py /app/app.py
ADD roop2.py /app/roop2.py
ADD requirements.txt /app/requirements.txt

COPY ./models /app/models

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install -r /app/requirements.txt

CMD [ "gradio" , "app.py", "--port", "8000"]