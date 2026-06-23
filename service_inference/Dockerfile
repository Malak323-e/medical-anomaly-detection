FROM python:3.10-slim
WORKDIR /app
RUN pip install fastapi uvicorn pillow python-multipart torchvision --extra-index-url https://download.pytorch.org/whl/cpu torch --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]