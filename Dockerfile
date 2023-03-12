FROM python:3.10
WORKDIR /app
RUN apt update && apt install -y build-essential gcc-10 clang cmake cppcheck valgrind afl gcc-multilib libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . .
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]