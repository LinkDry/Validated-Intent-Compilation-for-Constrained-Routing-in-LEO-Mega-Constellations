"""OD traffic demand generation with multiple scenarios."""

import numpy as np
from typing import Optional

# Major city hotspot coordinates (lat, lon)
HOTSPOTS = {
    "new_york": (40.7, -74.0),
    "london": (51.5, -0.1),
    "tokyo": (35.7, 139.7),
    "shanghai": (31.2, 121.5),
    "mumbai": (19.1, 72.9),
    "sao_paulo": (-23.5, -46.6),
    "sydney": (-33.9, 151.2),
    "dubai": (25.2, 55.3),
}

# Coverage radius in degrees (approx 1500km ground footprint)
HOTSPOT_RADIUS_DEG = 15.0


class TrafficDemandGenerator:
    """Generates OD demand matrices for LEO routing scenarios.

    Each scenario returns a sparse demand matrix [N, N] where
    demand[i][j] = number of packets from sat i to sat j per epoch.
    """

    def __init__(self, num_sats: int, seed: int = 42):
        self.num_sats = num_sats
        self.rng = np.random.RandomState(seed)

    def _sats_near_point(
        self, latlon: np.ndarray, target_lat: float, target_lon: float,
        radius_deg: float = HOTSPOT_RADIUS_DEG,
    ) -> np.ndarray:
        """Return indices of satellites within radius of a ground point."""
        dlat = latlon[:, 0] - target_lat
        dlon = latlon[:, 1] - target_lon
        # Wrap longitude
        dlon = np.where(dlon > 180, dlon - 360, dlon)
        dlon = np.where(dlon < -180, dlon + 360, dlon)
        dist = np.sqrt(dlat ** 2 + dlon ** 2)
        return np.where(dist < radius_deg)[0]

    def uniform(
        self, latlon: np.ndarray, lam: float = 2.0,
        sparsity: float = 0.02,
    ) -> np.ndarray:
        """Uniform low demand across random OD pairs."""
        N = self.num_sats
        demand = np.zeros((N, N), dtype=np.float32)
        # Only populate a fraction of OD pairs
        n_pairs = int(N * N * sparsity)
        srcs = self.rng.randint(0, N, n_pairs)
        dsts = self.rng.randint(0, N, n_pairs)
        valid = srcs != dsts
        demand[srcs[valid], dsts[valid]] = self.rng.poisson(lam, valid.sum()).astype(np.float32)
        return demand

    def hotspot(
        self, latlon: np.ndarray, lam_hot: float = 20.0,
        lam_bg: float = 1.0, n_hotspots: int = 5,
    ) -> np.ndarray:
        """High demand between satellites covering major cities."""
        N = self.num_sats
        demand = np.zeros((N, N), dtype=np.float32)

        # Select hotspot cities
        cities = list(HOTSPOTS.values())
        selected = [cities[i % len(cities)] for i in range(n_hotspots)]

        # Find sats near each hotspot
        hotspot_sats = set()
        for lat, lon in selected:
            near = self._sats_near_point(latlon, lat, lon)
            hotspot_sats.update(near.tolist())

        hotspot_sats = list(hotspot_sats)
        if len(hotspot_sats) < 2:
            return self.uniform(latlon)

        # High demand between hotspot sats
        for s in hotspot_sats:
            for d in hotspot_sats:
                if s != d:
                    demand[s, d] = self.rng.poisson(lam_hot)

        # Low background demand
        n_bg = int(N * N * 0.005)
        srcs = self.rng.randint(0, N, n_bg)
        dsts = self.rng.randint(0, N, n_bg)
        valid = srcs != dsts
        demand[srcs[valid], dsts[valid]] += self.rng.poisson(lam_bg, valid.sum())

        return demand.astype(np.float32)

    def regional(
        self, latlon: np.ndarray,
        lat_range: tuple = (20.0, 50.0),
        lon_range: tuple = (100.0, 150.0),
        lam_regional: float = 15.0,
        lam_bg: float = 1.0,
    ) -> np.ndarray:
        """Heavy traffic within a geographic region (default: Asia-Pacific)."""
        N = self.num_sats
        demand = np.zeros((N, N), dtype=np.float32)

        in_region = (
            (latlon[:, 0] >= lat_range[0]) & (latlon[:, 0] <= lat_range[1])
            & (latlon[:, 1] >= lon_range[0]) & (latlon[:, 1] <= lon_range[1])
        )
        regional_sats = np.where(in_region)[0]

        if len(regional_sats) < 2:
            return self.uniform(latlon)

        for s in regional_sats:
            for d in regional_sats:
                if s != d:
                    demand[s, d] = self.rng.poisson(lam_regional)

        # Background
        n_bg = int(N * N * 0.005)
        srcs = self.rng.randint(0, N, n_bg)
        dsts = self.rng.randint(0, N, n_bg)
        valid = srcs != dsts
        demand[srcs[valid], dsts[valid]] += self.rng.poisson(lam_bg, valid.sum())

        return demand.astype(np.float32)

    def polar_stress(
        self, latlon: np.ndarray, lam: float = 15.0,
    ) -> np.ndarray:
        """OD pairs that must transit through polar region.

        Routes between North America and Northern Europe force
        traffic through high-latitude paths.
        """
        N = self.num_sats
        demand = np.zeros((N, N), dtype=np.float32)

        na_sats = self._sats_near_point(latlon, 45.0, -90.0, 20.0)
        eu_sats = self._sats_near_point(latlon, 55.0, 15.0, 20.0)

        if len(na_sats) == 0 or len(eu_sats) == 0:
            return self.uniform(latlon)

        for s in na_sats:
            for d in eu_sats:
                demand[s, d] = self.rng.poisson(lam)
            for d in na_sats:
                if s != d:
                    demand[s, d] = self.rng.poisson(lam / 3)

        for s in eu_sats:
            for d in na_sats:
                demand[s, d] = self.rng.poisson(lam)

        return demand.astype(np.float32)

    def flash(
        self, latlon: np.ndarray,
        n_flash_nodes: int = 15,
        lam_flash: float = 30.0,
        lam_bg: float = 2.0,
    ) -> np.ndarray:
        """Sudden demand spike at random nodes (disaster/event)."""
        N = self.num_sats
        demand = self.uniform(latlon, lam=lam_bg)

        flash_nodes = self.rng.choice(N, n_flash_nodes, replace=False)
        for s in flash_nodes:
            targets = self.rng.choice(N, 20, replace=False)
            for d in targets:
                if s != d:
                    demand[s, d] += self.rng.poisson(lam_flash)

        return demand.astype(np.float32)

    def generate(
        self, scenario: str, latlon: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Generate demand matrix for a named scenario."""
        generators = {
            "uniform": self.uniform,
            "hotspot": self.hotspot,
            "regional": self.regional,
            "polar_stress": self.polar_stress,
            "flash": self.flash,
        }
        if scenario not in generators:
            raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(generators.keys())}")
        return generators[scenario](latlon, **kwargs)

    # Training scenarios (used during training)
    TRAIN_SCENARIOS = ["uniform", "hotspot", "regional"]
    # Held-out scenarios (only for evaluation)
    EVAL_SCENARIOS = ["polar_stress", "flash"]
