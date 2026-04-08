"""Train GNN routing policy with PPO."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import yaml
import torch
import numpy as np

from env.routing_env import LEORoutingEnv
from policy.routing_policy import GNNRoutingPolicy
from policy.ppo import PPO
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-episodes", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pretrained", default=None, help="Path to pretrained.pt for warm start")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed or cfg["training"]["seed"]
    total_episodes = args.total_episodes or cfg["training"]["total_episodes"]
    output_dir = args.output_dir or cfg["training"]["output_dir"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {seed}")

    env = LEORoutingEnv(
        num_planes=cfg["constellation"]["num_planes"],
        sats_per_plane=cfg["constellation"]["sats_per_plane"],
        altitude_km=cfg["constellation"]["altitude_km"],
        inclination_deg=cfg["constellation"]["inclination_deg"],
        max_isl_range_km=cfg["isl"]["max_range_km"],
        polar_lat_threshold_deg=cfg["isl"]["polar_lat_threshold_deg"],
        max_queue=cfg["env"]["max_queue"],
        max_hops=cfg["env"]["max_hops"],
        scenario=cfg["env"]["scenario"],
        seed=seed,
        w_delivery=cfg["env"]["w_delivery"],
        w_latency=cfg["env"]["w_latency"],
        w_drop=cfg["env"]["w_drop"],
        w_balance=cfg["env"]["w_balance"],
        max_acceptable_delay_ms=cfg["env"]["max_acceptable_delay_ms"],
    )

    policy = GNNRoutingPolicy(
        node_feat_dim=env.node_feat_dim,
        edge_feat_dim=env.edge_feat_dim,
        hidden_dim=cfg["policy"]["hidden_dim"],
        max_degree=env.max_degree,
    )
    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        print(f"Loaded pretrained weights from {args.pretrained}")
        if "pretrain_acc" in ckpt:
            print(f"  Pretrain acc={ckpt['pretrain_acc']:.3f}, PDR={ckpt.get('pretrain_pdr', 'N/A')}")

    ppo = PPO(
        policy,
        lr=cfg["ppo"]["lr"],
        gamma=cfg["ppo"]["gamma"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        clip_eps=cfg["ppo"]["clip_eps"],
        entropy_coef=cfg["ppo"]["entropy_coef"],
        value_coef=cfg["ppo"]["value_coef"],
        max_grad_norm=cfg["ppo"]["max_grad_norm"],
        n_epochs=cfg["ppo"]["n_epochs"],
        batch_size=cfg["ppo"]["batch_size"],
        device=device,
    )

    trainer = Trainer(
        env=env,
        ppo=ppo,
        output_dir=output_dir,
        eval_interval=cfg["training"]["eval_interval"],
        eval_episodes=cfg["training"]["eval_episodes"],
        rollout_steps=cfg["training"]["rollout_steps"],
        log_interval=cfg["training"]["log_interval"],
    )

    trainer.train(total_episodes)


if __name__ == "__main__":
    main()
