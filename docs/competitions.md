# Omron Competitions Framework

🥩 **Competitions Overview**
Omron's competition system drives innovation in zkML performance by creating structured challenges where miners optimize circuits for specific tasks. Competitions leverage Omron's Proof-of-Inference mechanism([1](https://docs.omron.ai/intro-to-omron)) to verify miner submissions while maintaining computational integrity.

## Competition Lifecycle

### Phases

1. **Pending**

   - Competition scheduled but not yet active
   - Baseline model and evaluation criteria published
   - Miners prepare circuits using competition template

2. **Active**

   - Open submission period (typically 7-14 days)
   - Real-time leaderboard tracking
   - Continuous circuit evaluation

3. **Completed**

   - Final scores calculated
   - Rewards distributed
   - SOTA circuit preserved for future benchmarking

4. **Inactive**
   - Between competition periods
   - Historical data analysis
   - Preparation for next challenge

## Miner Participation

### Requirements

- Submit optimized zk-circuits matching competition template
- Meet minimum hardware specs([2](https://docs.omron.ai/miner-validator-resources))

### Submission Process

1. Clone competition circuit template
2. Optimize model while maintaining I/O spec
3. Generate verification artifacts:
   - `settings.json
   - `model.compiled`
   - `vk.key`
   - `pk.key`
4. Commit circuit to the blockchain

## Evaluation Criteria

| Metric        | Weight | Description                         |
| ------------- | ------ | ----------------------------------- |
| Accuracy      | 40%    | Output similarity vs baseline model |
| Proof Size    | 30%    | Byte size of generated ZK proofs    |
| Response Time | 30%    | End-to-end proof generation latency |

> [!CAUTION]
> Should any proofs fail to verify, the miner will be assigned a zero score.

## Competition Management

### Key Features

- Automatic state transitions
- Real-time metrics tracking
- Fraud detection via proof verification

### Monitoring

- WandB integration for performance tracking
- Public leaderboard on Omron dashboard
- Daily score snapshots

## Risks

- Invalid proofs result in immediate disqualification
- Late submissions not accepted after end timestamp

**Links**
[Technical Roadmap](https://docs.omron.ai/technical-roadmap) • [Miner Setup](https://docs.omron.ai/miner-validator-resources) • [zkML Documentation](https://docs.omron.ai/custom_circuit_integrations)
