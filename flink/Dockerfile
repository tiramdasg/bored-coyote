FROM flink:1.17.1-java11

# install Python and set up aliases
# update indices
RUN apt-get update && \
    # install basic necessities & dependencies, -y -> "answer yes, if asked to allocate extra space on dependency installation" \
    # we install all optional dependencies, because flink does not disclose a list and its really annoying to restart the entire process to find out what dependencies are missing :)
    apt-get install -y libssl-dev openssl build-essential gdb lcov pkg-config \
      libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
      libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
      lzma lzma-dev tk-dev uuid-dev zlib1g-dev unzip && \
    # download python 3.10.14 source - version needed for tf-federated
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    # unpack python 3.10.14
    tar xzvf Python-3.10.14.tgz && \
    # configure and install python
    cd Python-3.10.14 && \
    ./configure --with-pydebug && \
    make && \
    make install && \
    # introduce soft link "python" for python3. this is necessary due to flink.
    ln -s /usr/local/bin/python3 /usr/bin/python \

# beware: if we do not do ML within flink, we can remove tf. etc. from this requirements file.
ADD flink-req.txt .
RUN pip3 install --upgrade pip && pip3 install -r flink-req.txt
RUN pip3 install apache-flink==1.17.0

RUN wget -P /opt/flink/lib/ https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/1.17.1/flink-sql-connector-kafka-1.17.1.jar; \
    wget -P /opt/flink/lib/ https://repo.maven.apache.org/maven2/org/apache/kafka/kafka-clients/3.2.3/kafka-clients-3.2.3.jar; \
    wget -P /opt/flink/lib/ https://github.com/knaufk/flink-faker/releases/download/v0.5.3/flink-faker-0.5.3.jar

# running 'typical'
CMD ["/docker-entrypoint.sh"]

