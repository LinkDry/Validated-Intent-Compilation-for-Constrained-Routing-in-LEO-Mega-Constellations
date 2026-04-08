"""Test baselines on the routing environment."""
from env.routing_env import LEORoutingEnv
from baselines.shortest_path import DijkstraRouter, RandomRouter
import numpy as np
import time

env = LEORoutingEnv(scenario="uniform", seed=42)

# --- Dijkstra (per-destination next-hop) ---
router = DijkstraRouter()
obs, info = env.reset()
total_pdr = []
total_reward = []
t0 = time.time()
n_steps = 10

for step in range(n_steps):
    table = router.build_nexthop_table(obs)
    env.set_nexthop_table(table)
    action = router.select_action(obs)  # dummy, env uses table
    obs, reward, terminated, truncated, info = env.step(action)
    total_pdr.append(info["pdr"])
    total_reward.append(reward)
    if terminated:
        break

elapsed = time.time() - t0
print(f"{'Dijkstra':10s}: PDR={np.mean(total_pdr):.3f}, reward={np.mean(total_reward):.4f}, "
      f"time={elapsed:.1f}s ({elapsed/n_steps:.2f}s/step)")
env.clear_nexthop_table()

# --- Random ---
obs, info = env.reset()
total_pdr = []
total_reward = []
t0 = time.time()

for step in range(n_steps):
    action = RandomRouter().select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_pdr.append(info["pdr"])
    total_reward.append(reward)
    if terminated:
        break

elapsed = time.time() - t0
print(f"{'Random':10s}: PDR={np.mean(total_pdr):.3f}, reward={np.mean(total_reward):.4f}, "
      f"time={elapsed:.1f}s ({elapsed/n_steps:.2f}s/step)")

print("\nBaseline test PASSED")
