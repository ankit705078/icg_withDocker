FROM ubuntu:latest
MAINTAINER "Ankit Yadav"
RUN apt update -y && apt install python3-pip -y
RUN mkdir icgapp
COPY Main_dir/ /icgapp/
WORKDIR /icgapp
RUN ls -la /icgapp/*

RUN pip3 --no-cache-dir install -r requirements.txt
#EXPOSE 5000

#ENTRYPOINT ["python3"]
#CMD ["main.py"]
CMD gunicorn main:app --bind 0.0.0.0:$PORT --reload