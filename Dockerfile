# Etapa 1: Construcción
FROM python:3.11-alpine as builder
WORKDIR /avs
RUN apk add --no-cache gcc musl-dev python3-dev
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Etapa 2: Imagen final
FROM python:3.11-alpine
WORKDIR /avs
COPY --from=builder /install /usr/local
COPY . .
CMD ["flask", "run", "--host=0.0.0.0"]
