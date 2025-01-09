NETUID ?= 2
WALLET_NAME ?= default
WALLET_HOTKEY ?= default
WALLET_PATH ?= $(HOME)/.bittensor
MINER_PORT ?= 8091
VALIDATOR_PORT ?= 8000

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
	docker stop omron-miner || true
	docker rm omron-miner || true
	docker run \
		--detach \
		--name omron-miner \
		-p $(MINER_PORT):8091 \
		-v $(WALLET_PATH):/root/.bittensor \
		omron miner.py \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(WALLET_HOTKEY) \
		--netuid $(NETUID)

validator:
	@echo "Using wallet path: $(WALLET_PATH)"
	docker stop omron-validator || true
	docker rm omron-validator || true
	docker run \
		--detach \
		--name omron-validator \
		-p $(VALIDATOR_PORT):8000 \
		-v $(WALLET_PATH):/root/.bittensor \
		omron validator.py \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(WALLET_HOTKEY) \
		--netuid $(NETUID)

test-miner:
	@echo "Using wallet path: $(WALLET_PATH)"
	docker stop omron-miner || true
	docker rm omron-miner || true
	docker run \
		--detach \
		--name omron-miner \
		-p $(MINER_PORT):8091 \
		-v $(WALLET_PATH):/root/.bittensor \
		omron miner.py \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(WALLET_HOTKEY) \
		--netuid 118 \
		--subtensor.network test \
		--disable-blacklist

test-validator:
	@echo "Using wallet path: $(WALLET_PATH)"
	docker stop omron-validator || true
	docker rm omron-validator || true
	docker run \
		--detach \
		--name omron-validator \
		-p $(VALIDATOR_PORT):8000 \
		-v $(WALLET_PATH):/root/.bittensor \
		omron validator.py \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(WALLET_HOTKEY) \
		--netuid 118 \
		--subtensor.network test
