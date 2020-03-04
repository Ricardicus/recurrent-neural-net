FROM ubuntu AS compile-image
RUN apt-get update
RUN apt-get -y install build-essential

# Preparing binary
WORKDIR /usr/src/app

COPY . .

RUN make

# Preparing node App
FROM ubuntu AS run-image
RUN apt-get update
RUN apt-get install nodejs -y 
RUN apt-get install npm -y
RUN apt-get install apt-utils -y

WORKDIR /usr/src/app

RUN mkdir /usr/src/app/data

COPY --from=compile-image /usr/src/app/net .
COPY --from=compile-image /usr/src/app/data/* /usr/src/app/data/

COPY ./docker/node/html .
COPY ./docker/node/*.js /usr/src/app/

EXPOSE 8080

CMD ["node", "app.js"]