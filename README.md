# Validated Intent Compilation for Constrained Routing in LEO Mega-Constellations

End-to-end system for intent-driven constrained routing in LEO satellite mega-constellations, combining a GNN cost-to-go router, an LLM intent compiler, and a 8-pass deterministic validator.

## Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| GNN Router | Packet Delivery Ratio | 99.8% (= Dijkstra) |
| GNN Router | Inference Speedup | 17× vs Dijkstra |
| LLM Compiler | Full Semantic Match | 86.2% (240 intents) |
| LLM Compiler | Compilation Rate | 97.9% |
| Validator | Corruption Detection | 100% (240 tests) |
| End-to-End | Constraint Violations | 0 (all scenarios) |
| OOD Generalization | Full Match | 81.8% (33 intents) |

## Architecture

```
Operator Intent (NL) → LLM Compiler → ConstraintProgram (JSON IR)
                                            ↓
                                    8-Pass Validator
                                            ↓
                                   Constraint Grounding
                                            ↓
                              GNN/Dijkstra Constrained Router
```

## Installation

```bash
# Clone
git clone https://github.com/[username]/leo-intent-routing.git
cd leo-intent-routing

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
│   ├── compiler.py           # LLM intent compiler (6-shot + repair loop)
│   ├── verifier.py           # 8-pass deterministic validator
│   ├── schema.py             # ConstraintProgram IR definition
│   ├── constrained_router.py # Constraint grounding + routing
│   ├── rule_based_parser.py  # Rule-based baseline parser
│   └── benchmark/            # Evaluation benchmarks
│       ├── benchmark_240.json          # Main 240-intent benchmark
│       ├── benchmark_ood_expanded.json # OOD paraphrase benchmark
│       └── benchmark_ood_paraphrases.json
├── policy/                   # GNN routing policy
├── training/                 # GNN training pipeline
├── env/                      # Constellation simulation environment
├── constellation/            # Orbital mechanics + topology
├── evaluation/               # Evaluation utilities
├── baselines/                # Dijkstra + random baselines
├── scripts/                  # Evaluation and analysis scripts
│   ├── eval_benchmark.py     # Main 240-intent evaluation
│   ├── eval_ablations.py     # Ablation study (4 configs)
│   ├── eval_e2e.py           # End-to-end constrained routing
│   ├── eval_ood_benchmark.py # OOD generalization evaluation
│   ├── eval_rule_based.py    # Rule-based baseline evaluation
│   ├── eval_topology_sweep.py # Topology degradation sweep
│   ├── eval_verifier_audit.py # Verifier corruption audit
│   ├── eval_adversarial_safety.py # Adversarial safety tests
│   └── reachability_separation.py # Reachability analysis
├── output/                   # Results and paper artifacts
│   ├── pretrain_ctg3/best.pt # Trained GNN checkpoint
│   ├── *.json                # All experimental results
│   └── paper_*.tex           # LaTeX paper sections
└── configs/                  # Training configurations
```

## Reproducing Results

### 1. GNN Router Training

```bash
# Train GNN cost-to-go router (200 epochs, ~2h on RTX 4060)
python scripts/train.py --config configs/ctg_v3.yaml

# Evaluate routing performance
python scripts/evaluate.py --checkpoint output/pretrain_ctg3/best.pt
```

### 2. Intent Compilation Benchmark

Requires LM Studio running with Qwen3.5-9B.

```bash
# Full 240-intent benchmark
python scripts/eval_benchmark.py

# Ablation study (4 configurations)
python scripts/eval_ablations.py

# Rule-based baseline comparison
python scripts/eval_rule_based.py

# OOD generalization
python scripts/eval_ood_benchmark.py
```

### 3. End-to-End Evaluation

```bash
# Constrained routing (5 scenarios × 3 seeds)
python scripts/eval_e2e.py

# Reachability separation analysis
python scripts/reachability_separation.py
```

### 4. Robustness Analysis

```bash
# Topology degradation sweep (9 severity levels)
python scripts/eval_topology_sweep.py

# Verifier corruption audit (8 types × 30 injections)
python scripts/eval_verifier_audit.py

# Adversarial safety tests (15 tests, 3 categories)
python scripts/eval_adversarial_safety.py
```

## Constellation Parameters

| Parameter | Value |
|-----------|-------|
| Orbital planes | 20 |
| Satellites/plane | 20 |
| Total nodes | 400 |
| Altitude | 550 km |
| Inclination | 53° |
| ISL neighbors | 4 (grid) |

## Benchmark Categories

| Category | N | Description |
|----------|---|-------------|
| Single | 80 | One constraint type |
| Compositional | 100 | 2-4 combined constraints |
| Conditional | 30 | Event-triggered constraints |
| Infeasible | 30 | Physically unrealizable |

## Citation

```bibtex
@inproceedings{[author]2027intent,
  title={Validated Intent Compilation for Constrained Routing in LEO Mega-Constellations},
  author={[Authors]},
  booktitle={Proc. IEEE INFOCOM},
  year={2027}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
