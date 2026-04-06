FROM node:22-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models
COPY --from=frontend-build /frontend/dist /usr/share/nginx/html

COPY nginx-space.conf.template /etc/nginx/nginx-space.conf.template
COPY start-space.sh /app/start-space.sh
RUN sed -i 's/\r$//' /app/start-space.sh && chmod +x /app/start-space.sh

ENV PORT=7860
EXPOSE 7860

CMD ["/app/start-space.sh"]