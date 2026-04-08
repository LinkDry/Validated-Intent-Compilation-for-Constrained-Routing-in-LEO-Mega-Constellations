"""LEO constellation routing environment with graph-structured observations."""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from constellation.walker import WalkerDeltaConstellation
from constellation.isl_topology import ISLTopologyManager
from env.traffic import TrafficDemandGenerator


class LEORoutingEnv(gym.Env):
    """Gymnasium env for LEO ISL packet routing.

    Action: [N, N] int64 next-hop table where action[node, dst] is the
    neighbor slot index (0..max_degree-1) for forwarding packets from
    node toward dst. The env converts slot indices to actual node IDs
    via the neighbor_table.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_planes: int = 20,
        sats_per_plane: int = 20,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        max_isl_range_km: float = 5016.0,
        polar_lat_threshold_deg: float = 75.0,
        max_queue: int = 1000,
        max_hops: int = 20,
        scenario: str = "uniform",
        seed: int = 42,
        w_delivery: float = 1.0,
        w_latency: float = 0.3,
        w_drop: float = 0.5,
        w_balance: float = 0.1,
        max_acceptable_delay_ms: float = 100.0,
    ):
        super().__init__()
        self.constellation = WalkerDeltaConstellation(
            num_planes, sats_per_plane, altitude_km, inclination_deg,
        )
        self.topo_manager = ISLTopologyManager(
            self.constellation, max_isl_range_km, polar_lat_threshold_deg,
        )
        self.traffic_gen = TrafficDemandGenerator(
            self.constellation.num_sats, seed=seed,
        )
        self.N = self.constellation.num_sats
        self.max_queue = max_queue
        self.max_hops = max_hops
        self.scenario = scenario
        self.max_degree = self.topo_manager.max_degree
        self.w_delivery = w_delivery
        self.w_latency = w_latency
        self.w_drop = w_drop
        self.w_balance = w_balance
        self.max_delay = max_acceptable_delay_ms

        # Action: [N, N] next-hop slot table
        self.action_space = spaces.Box(
            low=0, high=self.max_degree - 1,
            shape=(self.N, self.N), dtype=np.int64,
        )
        self.node_feat_dim = 8
        self.edge_feat_dim = 4
        max_edges = self.N * self.max_degree * 2
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                -np.inf, np.inf, shape=(self.N, self.node_feat_dim), dtype=np.float32
            ),
            "edge_index": spaces.Box(
                0, self.N, shape=(2, max_edges), dtype=np.int64
            ),
            "edge_attr": spaces.Box(
                -np.inf, np.inf, shape=(max_edges, self.edge_feat_dim), dtype=np.float32
            ),
            "num_edges": spaces.Discrete(max_edges + 1),
            "action_mask": spaces.Box(
                0, 1, shape=(self.N, self.max_degree), dtype=np.float32
            ),
            "neighbor_table": spaces.Box(
                -1, self.N, shape=(self.N, self.max_degree), dtype=np.int64
            ),
            "global_features": spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32
            ),
            "demand_matrix": spaces.Box(
                0, np.inf, shape=(self.N, self.N), dtype=np.float32
            ),
        })
        self.queues = np.zeros(self.N, dtype=np.float32)
        self.epoch = 0
        self.t_seconds = 0.0
        self._edge_index = None
        self._edge_attr = None
        self._adjacency = None
        self._neighbor_table = None
        self._action_mask = None
        self._latlon = None
        self._demand_cache = None
        self._edge_delays = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.traffic_gen.rng = np.random.RandomState(seed)
        self.queues = np.zeros(self.N, dtype=np.float32)
        self.epoch = 0
        self.t_seconds = 0.0
        self._update_topology()
        self._demand_cache = self.traffic_gen.generate(self.scenario, self._latlon)
        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action):
        """Step with [N, N] next-hop slot table."""
        metrics = self._forward_packets(action)
        self.epoch += 1
        self.t_seconds = self.epoch * self.constellation.epoch_duration_s
        self._update_topology()
        self._demand_cache = self.traffic_gen.generate(self.scenario, self._latlon)
        reward = self._compute_reward(metrics)
        terminated = self.epoch >= self.constellation.epochs_per_orbit
        obs = self._get_obs()
        info = self._get_info()
        info.update(metrics)
        return obs, reward, terminated, False, info

    def _update_topology(self):
        self._edge_index, self._edge_attr, self._adjacency = (
            self.topo_manager.compute_topology(self.t_seconds)
        )
        self._neighbor_table, self._action_mask = (
            self.topo_manager.get_neighbor_mask(self._adjacency)
        )
        self._latlon = self.constellation.get_latlon(self.t_seconds)
        ei = self._edge_index.numpy()
        ea = self._edge_attr.numpy()
        self._edge_delays = {}
        for idx in range(ei.shape[1]):
            self._edge_delays[(int(ei[0, idx]), int(ei[1, idx]))] = ea[idx, 0]

        # Build [N, K] neighbor delay table
        self._neighbor_delays = np.full(
            (self.N, self.max_degree), 0.0, dtype=np.float32,
        )
        for i in range(self.N):
            for k in range(self.max_degree):
                nbr = self._neighbor_table[i, k]
                if nbr >= 0:
                    self._neighbor_delays[i, k] = self._edge_delays.get((i, int(nbr)), 0.0)

    def _get_obs(self):
        ll = self._latlon
        demand = self._demand_cache
        total_demand = demand.sum(axis=1)
        total_sink = demand.sum(axis=0)

        avg_neigh_queue = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            valid = self._neighbor_table[i]
            valid = valid[valid >= 0]
            if len(valid) > 0:
                avg_neigh_queue[i] = self.queues[valid].mean()

        num_neighbors = self._action_mask.sum(axis=1)
        is_polar = (
            np.abs(ll[:, 0]) > self.topo_manager.polar_lat_threshold
        ).astype(np.float32)

        # Positional features: plane_id and sat_id within plane
        spp = self.constellation.sats_per_plane
        npl = self.constellation.num_planes
        plane_id = np.array([i // spp for i in range(self.N)], dtype=np.float32) / max(npl - 1, 1)
        sat_in_plane = np.array([i % spp for i in range(self.N)], dtype=np.float32) / max(spp - 1, 1)

        node_features = np.stack([
            self.queues / self.max_queue,
            total_demand / (total_demand.max() + 1e-8),
            total_sink / (total_sink.max() + 1e-8),
            ll[:, 0] / 90.0,
            ll[:, 1] / 180.0,
            num_neighbors / self.max_degree,
            plane_id,
            sat_in_plane,
        ], axis=-1).astype(np.float32)

        E = self._edge_index.shape[1]
        max_E = self.observation_space["edge_index"].shape[1]
        edge_index_pad = np.zeros((2, max_E), dtype=np.int64)
        edge_index_pad[:, :E] = self._edge_index.numpy()
        edge_attr_pad = np.zeros((max_E, self.edge_feat_dim), dtype=np.float32)
        edge_attr_pad[:E] = self._edge_attr.numpy()

        total_load = total_demand.sum()
        avg_queue = self.queues.mean()
        epoch_frac = self.epoch / self.constellation.epochs_per_orbit
        global_features = np.array(
            [total_load / (self.N * 10), avg_queue / self.max_queue, epoch_frac],
            dtype=np.float32,
        )

        return {
            "node_features": node_features,
            "edge_index": edge_index_pad,
            "edge_attr": edge_attr_pad,
            "num_edges": E,
            "action_mask": self._action_mask.copy(),
            "neighbor_table": self._neighbor_table.copy(),
            "neighbor_delays": self._neighbor_delays.copy(),
            "global_features": global_features,
            "demand_matrix": demand.copy(),
        }

    def _forward_packets(self, action):
        """Forward packets using [N, N] next-hop slot table.

        action[node, dst] = neighbor slot index -> neighbor_table[node, slot] = next-hop node ID
        """
        demand = self._demand_cache
        nt = self._neighbor_table
        edge_delays = self._edge_delays

        total_inj = 0
        total_del = 0
        total_drop = 0
        delays = []

        od_pairs = np.argwhere(demand > 0)
        for src, dst in od_pairs:
            n_pkt = int(demand[src, dst])
            total_inj += n_pkt
            current = int(src)
            dst_int = int(dst)
            path_delay = 0.0
            delivered = False
            visited = set()

            for hop in range(self.max_hops):
                if current == dst_int:
                    delivered = True
                    break

                if current in visited:
                    break
                visited.add(current)

                # Per-destination next-hop lookup
                slot = int(action[current, dst_int])
                next_node = int(nt[current, slot])

                if next_node < 0:
                    break

                ld = edge_delays.get((current, next_node))
                if ld is None:
                    break
                if self.queues[next_node] + n_pkt > self.max_queue:
                    break

                path_delay += ld
                current = next_node

            if delivered:
                total_del += n_pkt
                delays.extend([path_delay] * n_pkt)
            else:
                total_drop += n_pkt

        # Queue dynamics
        service_rate = 50.0
        self.queues = np.maximum(0, self.queues - service_rate)
        for src, dst in od_pairs:
            n_pkt = int(demand[src, dst])
            self.queues[int(src)] = min(
                self.queues[int(src)] + n_pkt * 0.1, self.max_queue
            )

        pdr = total_del / (total_inj + 1e-8)
        drop_rate = total_drop / (total_inj + 1e-8)
        mean_delay = np.mean(delays) if delays else 0.0

        return {
            "pdr": pdr,
            "drop_rate": drop_rate,
            "mean_delay_ms": mean_delay,
            "total_injected": total_inj,
            "total_delivered": total_del,
            "total_dropped": total_drop,
            "link_utilization_balance": 1.0,
        }

    def _compute_reward(self, m):
        return (
            self.w_delivery * m["pdr"]
            - self.w_latency * (m["mean_delay_ms"] / self.max_delay)
            - self.w_drop * m["drop_rate"]
            + self.w_balance * m["link_utilization_balance"]
        )

    def _get_info(self):
        return {
            "epoch": self.epoch,
            "t_seconds": self.t_seconds,
            "mean_queue": float(self.queues.mean()),
            "max_queue": float(self.queues.max()),
            "scenario": self.scenario,
        }
