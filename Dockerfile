# 베이스 이미지
FROM python:3.13-slim

# 작업 디렉토리 설정
WORKDIR /app

# OS 패키지 설치 (빌드 도구 + 필수 라이브러리)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    cmake \ 
    libffi-dev \
    libssl-dev \
    libpq-dev \
    && apt-get clean

# pip, setuptools, wheel 업데이트
RUN pip install --upgrade pip setuptools wheel

# requirements.txt 복사 및 설치
COPY requirements.txt .

# numpy 먼저 설치
RUN pip install numpy

# 그 외 requirements 설치
RUN pip install -r requirements.txt

# 앱 소스 코드 복사
COPY . .

# 외부에서 접속 가능하도록 uvicorn 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]