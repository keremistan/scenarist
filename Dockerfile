FROM python:3.13-slim
LABEL authors="keremdede"

RUN echo "setting the working dir"
WORKDIR /usr/src

RUN echo "copying the requirements"
COPY requirements.txt ./

RUN echo "copying the fastapi "
COPY app/ ./app

RUN echo "copying the streamlit"
COPY ui/ ./ui

RUN echo "copying the .env"
COPY .env ./

RUN echo "installing the requirements"
RUN pip3 install --no-cache-dir -r ./requirements.txt

#RUN echo "running fastapi dev command on api/main.py"
#CMD ["fastapi", "dev", "api/main.py"]
