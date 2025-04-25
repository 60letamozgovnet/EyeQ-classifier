@echo off
setlocal

echo 🚀 Старт Docker-сборки...
docker build -t eyeq-model .

echo 📦 Запуск модели...
docker run --rm -v "%cd%\main":/app/main eyeq-model

endlocal
pause