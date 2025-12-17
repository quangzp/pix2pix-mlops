FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
ENTRYPOINT ["python", "src/train.py"]