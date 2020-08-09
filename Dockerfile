FROM continuumio/anaconda3:4.4.0
MAINTAINER abhinavanand1905@gmail.com
COPY ./flask_model /usr/local/python
EXPOSE 5000
WORKDIR /usr/local/python
RUN pip install -r requirements.txt
CMD python app.py
