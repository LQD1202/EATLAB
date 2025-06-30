FROM python:3.10-slim

# 1. Cài gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Cài thư viện Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4. Copy toàn bộ mã nguồn
COPY . .

# 5. Command chạy ứng dụng
CMD ["python", "inference.py"]
