"""Test GNN policy forward pass and PPO action selection."""
import torch
from env.routing_env import LEORoutingEnv
from policy.routing_policy import GNNRoutingPolicy
from policy.ppo import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create env and policy
env = LEORoutingEnv(scenario="uniform", seed=42)
policy = GNNRoutingPolicy(
    node_feat_dim=env.node_feat_dim,
    edge_feat_dim=env.edge_feat_dim,
    hidden_dim=64,
    max_degree=env.max_degree,
)
print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")

# Test forward pass
obs, info = env.reset()
data, mask = GNNRoutingPolicy.obs_to_pyg(obs, device)
print(f"PyG Data: x={data.x.shape}, edge_index={data.edge_index.shape}, edge_attr={data.edge_attr.shape}")

policy.to(device)
policy.eval()
with torch.no_grad():
    logits, value = policy(data, mask)
    print(f"Logits: {logits.shape}, Value: {value.shape} = {value.item():.4f}")

# Test action sampling
with torch.no_grad():
    action, log_prob, entropy, value = policy.get_action_and_value(data, mask)
    print(f"Action: {action.shape}, log_prob={log_prob.item():.4f}, entropy={entropy.item():.4f}")

# Test PPO integration
ppo = PPO(policy, device=device, lr=3e-4, batch_size=4)
obs, info = env.reset()

# Collect a few steps
for step in range(8):
    action, log_prob, value = ppo.select_action(obs)
    obs_next, reward, terminated, truncated, info = env.step(action)
    ppo.store_transition(obs, action, log_prob, reward, value, terminated)
    obs = obs_next
    if terminated:
        obs, info = env.reset()

# Run PPO update
update_stats = ppo.update()
print(f"\nPPO update: {update_stats}")

# Save/load test
ppo.save("/tmp/test_policy.pt")
ppo.load("/tmp/test_policy.pt")
print("\nSave/load OK")
print("\nPolicy test PASSED")
