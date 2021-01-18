FROM ubuntu:18.04

RUN apt-get update && apt-get -y install build-essential

WORKDIR /usr/src/app

COPY . .

RUN make

CMD ["./net", "data/eddan_full.txt", "-st", "1000", "-ep", "40000"]
ENTRYPOINT ["./net"]
