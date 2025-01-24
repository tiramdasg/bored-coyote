FROM python:3.10

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "/src/start_container.py"]
