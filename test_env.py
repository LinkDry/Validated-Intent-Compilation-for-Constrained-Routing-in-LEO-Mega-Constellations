"""Quick test for the routing environment."""
from env.routing_env import LEORoutingEnv
import numpy as np

env = LEORoutingEnv(scenario="uniform", seed=42)
obs, info = env.reset()
print("Reset OK")
print(f"  node_features: {obs['node_features'].shape}")
print(f"  edge_index: {obs['edge_index'].shape}, num_edges: {obs['num_edges']}")
print(f"  action_mask: {obs['action_mask'].shape}")
print(f"  global_features: {obs['global_features']}")

# Random action respecting mask
def random_action(obs, N):
    action = np.zeros(N, dtype=np.int64)
    for i in range(N):
        valid = np.where(obs["action_mask"][i] > 0)[0]
        if len(valid) > 0:
            action[i] = np.random.choice(valid)
    return action

action = random_action(obs, env.N)
obs2, reward, terminated, truncated, info2 = env.step(action)
print(f"\nStep 1:")
print(f"  reward={reward:.4f}")
print(f"  pdr={info2['pdr']:.3f}, drop={info2['drop_rate']:.3f}, delay={info2['mean_delay_ms']:.1f}ms")
print(f"  injected={info2['total_injected']}, delivered={info2['total_delivered']}, dropped={info2['total_dropped']}")
print(f"  terminated={terminated}")

# Run a few more steps
for i in range(5):
    action = random_action(obs2, env.N)
    obs2, reward, terminated, truncated, info2 = env.step(action)

print(f"\nAfter 6 steps: epoch={info2['epoch']}, reward={reward:.4f}, pdr={info2['pdr']:.3f}")
print("\nEnv test PASSED")
