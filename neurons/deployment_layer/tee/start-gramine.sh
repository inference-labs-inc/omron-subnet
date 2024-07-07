export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-bigdl-llm-gramine-ref:2.4.0-SNAPSHOT
export DOCKER_NAME=fastchat-gramine
docker run -itd \
        --privileged \
        --net=host \
        --name="${DOCKER_NAME}" \
        --cpuset-cpus="0-47" \
         -v /mnt/sde/tpch-data:/llama \
        --shm-size="16g" \
        --memory="64g" \
        -e LOCAL_IP="${LOCAL_IP"} \
        -v "${NFS_INPUT_PATH"}:/llama \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        "${DOCKER_IMAGE"} bash
