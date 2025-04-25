#!/bin/bash
set -e

echo "🚀 Старт Docker-сборки..."
docker build -t eyeq-model .

echo "📦 Запуск модели..."
docker run --rm -v $(pwd)/main:/app/main eyeq-model