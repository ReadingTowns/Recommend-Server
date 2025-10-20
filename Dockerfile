# 베이스 이미지
FROM python:3.13-slim

# 작업 디렉토리 설정
WORKDIR /app

# OS 패키지 설치 (빌드 도구 + 필수 라이브러리)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    cmake \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/*

# pip, setuptools, wheel 업데이트
RUN pip install --upgrade pip setuptools wheel

# CPU 전용 PyTorch 설치
RUN pip install --no-cache-dir torch==2.9.0+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# sentence-transformers 설치
RUN pip install --no-cache-dir sentence-transformers

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 코드 복사
COPY . .

# 외부에서 접속 가능하도록 uvicorn 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]