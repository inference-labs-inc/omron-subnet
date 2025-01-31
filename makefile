OS := $(shell uname)
NETUID ?= 2
WALLET_NAME ?= default
WALLET_HOTKEY ?= default
WALLET_PATH ?= $(HOME)/.bittensor
ifeq ($(OS),Darwin)
    PUID ?= $(shell stat -f %u $(WALLET_PATH))
else
    PUID ?= $(shell stat -c %u $(WALLET_PATH))
endif
MINER_PORT ?= 8091
VALIDATOR_PORT ?= 8443

.PHONY: build stop clean miner-logs validator-logs miner validator test-miner test-validator

build:
	docker build -t omron -f Dockerfile .

stop:
	docker stop omron-miner || true
	docker stop omron-validator || true

clean:
	docker stop omron-miner || true
	docker stop omron-validator || true
	docker rm omron-miner || true
	docker rm omron-validator || true
	docker image rm omron || true
	docker image prune -f

miner-logs:
	docker logs -f omron-miner

validator-logs:
	docker logs -f omron-validator

miner:
	@echo "Using wallet path: $(WALLET_PATH)"
	@echo "Setting PUID to $(PUID)"
	docker stop omron-miner || true
	docker rm omron-miner || true
	docker run \
		--detach \
		--name omron-miner \
		-p $(MINER_PORT):8091 \
		-v $(WALLET_PATH):/home/ubuntu/.bittensor \
		-e PUID=$(PUID) \
		omron miner.py \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(WALLET_HOTKEY) \
		--netuid $(NETUID)

validator:
	@echo "Using wallet path: $(WALLET_PATH)"
	@echo "Setting PUID to $(PUID)"
	docker stop omron-validator || true
	docker rm omron-validator || true
	docker run \
		--detach \
		--name omron-validator \
		-p $(VALIDATOR_PORT):8443 \
		-v $(WALLET_PATH):/home/ubuntu/.bittensor \
		-e PUID=$(PUID) \
		omron validator.py \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(WALLET_HOTKEY) \
		--netuid $(NETUID)

test-miner:
	@echo "Using wallet path: $(WALLET_PATH)"
	@echo "Setting PUID to $(PUID)"
	docker stop omron-miner || true
	docker rm omron-miner || true
	docker run \
		--detach \
		--name omron-miner \
		-p $(MINER_PORT):8091 \
		-v $(WALLET_PATH):/home/ubuntu/.bittensor \
		-e PUID=$(PUID) \
		omron miner.py \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(WALLET_HOTKEY) \
		--netuid 118 \
		--subtensor.network test \
		--disable-blacklist

test-validator:
	@echo "Using wallet path: $(WALLET_PATH)"
	@echo "Setting PUID to $(PUID)"
	docker stop omron-validator || true
	docker rm omron-validator || true
	docker run \
		--detach \
		--name omron-validator \
		-p $(VALIDATOR_PORT):8443 \
		-v $(WALLET_PATH):/home/ubuntu/.bittensor \
		-e PUID=$(PUID) \
		omron validator.py \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(WALLET_HOTKEY) \
		--netuid 118 \
		--subtensor.network test

pm2-stop:
	pm2 stop omron-miner || true
	pm2 stop omron-validator || true

pm2-miner:
	uv sync --locked --no-dev
	cd neurons; \
	pm2 start miner.py --name omron-miner --interpreter ../.venv/bin/python --kill-timeout 3000 -- \
	--wallet.path $(WALLET_PATH) \
	--wallet.name $(WALLET_NAME) \
	--wallet.hotkey $(WALLET_HOTKEY) \
	--netuid $(NETUID)

pm2-validator:
	uv sync --locked --no-dev
	cd neurons; \
	pm2 start validator.py --name omron-validator --interpreter ../.venv/bin/python --kill-timeout 3000 -- \
	--wallet.path $(WALLET_PATH) \
	--wallet.name $(WALLET_NAME) \
	--wallet.hotkey $(WALLET_HOTKEY) \
	--netuid $(NETUID)

pm2-test-miner:
	uv sync --locked --no-dev
	cd neurons; \
	pm2 start miner.py --name omron-miner --interpreter ../.venv/bin/python --kill-timeout 3000 -- \
	--wallet.path $(WALLET_PATH) \
	--wallet.name $(WALLET_NAME) \
	--wallet.hotkey $(WALLET_HOTKEY) \
	--netuid 118 \
	--subtensor.network test \
	--disable-blacklist

pm2-test-validator:
	uv sync --locked --no-dev
	cd neurons; \
	pm2 start validator.py --name omron-validator --interpreter ../.venv/bin/python --kill-timeout 3000 -- \
	--wallet.path $(WALLET_PATH) \
	--wallet.name $(WALLET_NAME) \
	--wallet.hotkey $(WALLET_HOTKEY) \
	--netuid 118 \
	--subtensor.network test
