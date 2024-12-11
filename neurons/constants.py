# The model ID for a batched proof of weights model
BATCHED_PROOF_OF_WEIGHTS_MODEL_ID = (
    "e84b2e5f223621fa20078eb9f920d8d4d3a4ff95fa6e2357646fdbb43a2557c9"
)
# The model ID for a single proof of weights model
SINGLE_PROOF_OF_WEIGHTS_MODEL_ID = (
    "a849500803abdbb86a9460e18684a6411dc7ae0b75f1f6330e3028081a497dea"
)
# The model ID for a single proof of weights model, using the Jolt proof system
SINGLE_PROOF_OF_WEIGHTS_MODEL_ID_JOLT = (
    "37320fc74fec80805eedc8e92baf3c58842a2cb2a4ae127ad6e930f0c8441c7a"
)
IGNORED_MODEL_HASHES = [
    "0",
    "0a92bc32ea02abe54159da70aeb541d52c3cba27c8708669eda634e096a86f8b",
    "b7d33e7c19360c042d94c5a7360d7dc68c36dd56c449f7c49164a0098769c01f",
    "55de10a6bcf638af4bc79901d63204a9e5b1c6534670aa03010bae6045e3d0e8",
    "9998a12b8194d3e57d332b484ede57c3d871d42a176456c4e10da2995791d181",
    "ed8ba401d709ee31f6b9272163c71451da171c7d71800313fe5db58d0f6c483a",
]

# The maximum timespan allowed for miners to respond to a query
VALIDATOR_REQUEST_TIMEOUT_SECONDS = 120
# The timeout for aggregation requests
VALIDATOR_AGG_REQUEST_TIMEOUT_SECONDS = 600
# Maximum number of concurrent requests that the validator will handle
MAX_CONCURRENT_REQUESTS = 16
# Default proof size when we're unable to determine the actual size
DEFAULT_PROOF_SIZE = 5000
# Size in percent of the sample to be used for the maximum score median
MAXIMUM_SCORE_MEDIAN_SAMPLE = 0.05
# Shift in seconds to apply to the minimum response time for vertical asymptote adjustment
MINIMUM_SCORE_SHIFT = 0.0
# Weights version hyperparameter
WEIGHTS_VERSION = 1514
# Rate limit for weight updates
WEIGHT_RATE_LIMIT: int = 100
# Delay between requests
REQUEST_DELAY_SECONDS = 6
# Default maximum score
DEFAULT_MAX_SCORE = 1 / 235
# Default subnet UID
DEFAULT_NETUID = 2
# Validator stake threshold
VALIDATOR_STAKE_THRESHOLD = 1024
# 🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩🥩
STEAK = "🥩"
# Field modulus
FIELD_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)
# Whether on-chain proof of weights is enabled by default
ONCHAIN_PROOF_OF_WEIGHTS_ENABLED = False
# Frequency in terms of blocks at which proof of weights are posted
PROOF_OF_WEIGHTS_INTERVAL = 1000
# Maximum number of proofs to log at once
MAX_PROOFS_TO_LOG = 0
# Era period for proof of weights (mortality of the pow log)
PROOF_OF_WEIGHTS_LIFESPAN = 2
# Weights that determine the probability of selecting a circuit for benchmarking
CIRCUIT_WEIGHTS = {
    "1d60d545b7c5123fd60524dcbaf57081ca7dc4a9ec36c892927a3153328d17c0": 0,
    "33b92394b18412622adad75733a6fc659b4e202b01ee8a5465958a6bad8ded62": 0.20,
    "37320fc74fec80805eedc8e92baf3c58842a2cb2a4ae127ad6e930f0c8441c7a": 0,
    "a4bcecaf699fd9212600a1f2fcaa40c444e1aeaab409ea240a38c33ed356f4e2": 0.20,
    "a849500803abdbb86a9460e18684a6411dc7ae0b75f1f6330e3028081a497dea": 0.20,
    "e84b2e5f223621fa20078eb9f920d8d4d3a4ff95fa6e2357646fdbb43a2557c9": 0.20,
    "8dcff627a782525ea86196941a694ffbead179905f0cd4550ddc3df9e2b90924": 0.20,
}
# Maximum signature lifespan for WebSocket requests
MAX_SIGNATURE_LIFESPAN = 300
# Whitelisted public keys (ss58 addresses) we accept external requests from by default
# (even if an address is not in the metagraph)
WHITELISTED_PUBLIC_KEYS = []
