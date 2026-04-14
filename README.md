# LEON: LLM-Enabled Intent-Driven Operations for LEO Mega-Constellations

End-to-end framework for intent-driven constrained routing in LEO satellite mega-constellations, combining a GNN cost-to-go router, an LLM intent compiler, and an 8-pass deterministic validator with constructive feasibility certification.

**Paper:** *LEON: LLM-Enabled Intent-Driven Operations for LEO Mega-Constellations* (target: IEEE INFOCOM 2027)

## Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| GNN Router | Packet Delivery Ratio | 99.8% (= Dijkstra) |
| GNN Router | Inference Speedup | 17x vs Dijkstra |
| LLM Compiler | Full Semantic Match | 86.2% (240 intents, uniform random edge delays) |
| LLM Compiler | Compilation Rate | 97.9% (uniform random edge delays) |
| LLM Compiler | Compositional Advantage | +59.0pp over rule-based parsing |
| Validator | Unsafe Acceptance | 0% (8-pass pipeline, 30 infeasible intents) |
| Validator | Adversarial Detection | 100% (15 targeted attacks) |
| End-to-End | Constraint Violations | 0 (4 constrained scenarios) |
| OOD Generalization | Full Match | 81.8% (33 scorable paraphrases) |

## Architecture

```
Operator Intent (NL) --> LLM Compiler (Qwen3.5-9B, 5-shot + repair)
                                |
                         ConstraintProgram (JSON IR)
                                |
                      8-Pass Deterministic Validator
                       (Accept / Reject / Abstain)
                                |
                       Constraint Grounding
                                |
                    GNN / Dijkstra Constrained Router
```

## Installation

```bash
# Clone
git clone https://github.com/LinkDry/Validated-Intent-Compilation-for-Constrained-Routing-in-LEO-Mega-Constellations.git
cd Validated-Intent-Compilation-for-Constrained-Routing-in-LEO-Mega-Constellations

# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### LLM Setup (for intent compilation)

The LLM compiler requires a local LLM server (LM Studio or compatible OpenAI API):

1. Install [LM Studio](https://lmstudio.ai/)
2. Download Qwen3.5-9B (GGUF format)
3. Start the local server on `http://localhost:1234/v1`

## Project Structure

```
├── intent/                    # Intent compilation pipeline
│   ├── compiler.py           # LLM intent compiler (5-shot + repair loop)
│   ├── verifier.py           # 8-pass deterministic validator
│   ├── schema.py             # ConstraintProgram IR definition
│   ├── constrained_router.py # Constraint grounding + routing
│   ├── rule_based_parser.py  # Rule-based baseline parser
│   └── benchmark/            # Evaluation benchmarks
│       ├── benchmark_240.json          # Main 240-intent benchmark
│       ├── benchmark_ood_expanded.json # OOD expanded benchmark
│       └── benchmark_ood_paraphrases.json  # OOD paraphrase benchmark
├── constellation/            # Orbital mechanics + ISL topology
│   ├── walker.py             # Walker Delta constellation geometry
│   ├── isl_topology.py       # ISL topology (intra/inter-plane)
│   └── link_budget.py        # FSPL, delay, capacity models
├── policy/                   # GNN routing policy
│   ├── gat_encoder.py        # 3-layer GAT encoder
│   ├── cost_to_go.py         # MLP cost-to-go prediction head
│   └── routing_policy.py     # GNNRoutingPolicy (main policy)
├── env/                      # Constellation simulation environment
├── training/                 # GNN training pipeline
├── evaluation/               # Evaluation utilities
├── baselines/                # Dijkstra + random baselines
├── scripts/                  # All experiment entry points
├── output/                   # Results and model checkpoints
│   ├── pretrain_ctg3/best.pt # Trained GNN checkpoint
│   └── *.json                # All experimental results
├── paper/                    # LaTeX paper source
└── configs/                  # Training configurations
```

## Reproducing Results

### Important: Two Delay Models

The paper uses two edge delay models for different experiments. This is disclosed in the paper (Section IV-A) and affects reproducibility:

- **Uniform random delays** U(2.5, 15.0) ms: Used by `eval_ablations.py` (compiler ablation/comparison tables). Produces the 86.2% full match numbers.
- **Distance-based ISL delays** (haversine/300): Used by `eval_benchmark.py` (confusion matrix, E2E evaluation). Produces the 70.4% full match numbers.

The two runs also differ in coordinate discretization (endpoint-inclusive vs endpoint-exclusive). See the paper for details.

### 1. GNN Router Training

```bash
# Supervised pretraining from Dijkstra (200 epochs, ~2h on RTX 4060)
python scripts/pretrain.py --config configs/default.yaml

# Evaluate routing performance (5 traffic scenarios, 8 seeds)
python scripts/evaluate.py
```

### 2. Intent Compilation Benchmark

Requires LM Studio running with Qwen3.5-9B.

```bash
# Full 240-intent benchmark (distance-based delays, ~2h)
python scripts/eval_benchmark.py

# Ablation study: 6 configurations (uniform random delays, ~6h total)
python scripts/eval_ablations.py --all

# Rule-based baseline comparison
python scripts/eval_rule_based.py

# OOD generalization (38 paraphrased intents)
python scripts/eval_ood_benchmark.py
```

### 3. End-to-End Evaluation

```bash
# Constrained routing (5 scenarios x 3 seeds x 50 steps)
python scripts/eval_e2e.py

# Reachability separation analysis
python scripts/reachability_separation.py
```

### 4. Validator Analysis

```bash
# 3-way confusion matrix (Accept/Reject/Abstain)
python scripts/eval_confusion_matrix.py

# Independent Dijkstra oracle (32 routing-infeasible programs)
python scripts/eval_independent_oracle.py

# Adversarial safety tests (15 tests, 3 categories)
python scripts/eval_adversarial_safety.py

# Pass 8 runtime benchmark
python scripts/eval_pass8_runtime.py

# Verifier corruption audit (8 types x 30 injections)
python scripts/eval_verifier_audit.py
```

### 5. Robustness Analysis

```bash
# Topology degradation sweep (9 severity levels)
python scripts/eval_topology_sweep.py

# Cross-constellation generalization (3 configs)
python scripts/eval_cross_constellation.py

# Polar exclusion zone sweep (4 thresholds)
python scripts/eval_polar_exclusion.py

# Temporal degradation (stale routing tables)
python scripts/temporal_degradation.py
```

## Constellation Parameters

| Parameter | Value |
|-----------|-------|
| Orbital planes | 20 |
| Satellites/plane | 20 |
| Total nodes | 400 |
| Altitude | 550 km |
| Inclination | 53 deg |
| ISL neighbors | 4 (2 intra-plane + 2 inter-plane) |
| Polar dropout | >75 deg latitude (inter-plane ISLs) |

## Benchmark Categories

| Category | N | Description |
|----------|---|-------------|
| Single | 80 | One constraint type |
| Compositional | 100 | 2-4 combined constraints |
| Conditional | 30 | Event-triggered constraints |
| Infeasible | 30 | Physically unrealizable |

## Hardware

All experiments run on a single workstation:
- CPU: Intel Core Ultra 9 185H (16 GB RAM)
- GPU: NVIDIA RTX 4060 (8 GB VRAM)
- OS: Ubuntu 24.04 (WSL2)
- LLM: Qwen3.5-9B via LM Studio (local, no cloud API)

## Citation

```bibtex
@inproceedings{li2027leon,
  title={{LEON}: {LLM}-Enabled Intent-Driven Operations for {LEO} Mega-Constellations},
  author={Li, Yuanhang},
  booktitle={Proc. IEEE INFOCOM},
  year={2027}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
