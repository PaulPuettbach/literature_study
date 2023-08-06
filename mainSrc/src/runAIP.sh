#!/bin/bash

trap 'docker compose down' EXIT
trap 'docker compose down' INT TERM

cd ../$1
docker compose up --build