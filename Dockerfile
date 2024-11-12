FROM --platform=linux/amd64 ubuntu:noble

# Copy omron
COPY requirements.txt /opt/omron/requirements.txt
COPY __init__.py /opt/omron/__init__.py
COPY neurons /opt/omron/neurons

# Install dependencies
RUN apt update && \
    apt install -y \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    curl \
    make \
    clang \
    libssl-dev \
    llvm \
    libudev-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Use a venv because of https://peps.python.org/pep-0668/
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc
# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r /opt/omron/requirements.txt

WORKDIR /opt/omron/neurons
ENTRYPOINT ["/opt/venv/bin/python3"]
