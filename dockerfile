FROM python:3.9
RUN apt-get update
COPY src/ src/
COPY test/ test/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT streamlit run src/app_streamlit.py