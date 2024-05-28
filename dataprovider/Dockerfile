FROM python:3.10-slim

# add the kafka folder to the container
ADD subreddits.json .
ADD kafka-req.txt .
# install requirements
RUN pip3 install --upgrade pip && pip3 install -r kafka-req.txt

RUN mkdir -p /data
ENV DATA_DIR /data

# this way, changes to the code wont trigger reinstall of libraries in req if no changes in that layer
ADD DataProvider.py .

# run the data provider
CMD ["python3", "DataProvider.py"]