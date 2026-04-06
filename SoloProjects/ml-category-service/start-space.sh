#!/bin/sh
set -eu

PORT="${PORT:-7860}"
export PORT

envsubst '${PORT}' \
  < /etc/nginx/nginx-space.conf.template \
  > /etc/nginx/conf.d/default.conf

uvicorn app.main:app --host 127.0.0.1 --port 8000 &

exec nginx -g "daemon off;"
