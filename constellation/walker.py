"""Walker Delta constellation geometry and orbital mechanics."""

import numpy as np

# Constants
R_EARTH_KM = 6371.0
MU = 3.986004418e14  # Earth gravitational parameter (m^3/s^2)
OMEGA_EARTH = 7.2921159e-5  # Earth rotation rate (rad/s)


class WalkerDeltaConstellation:
    """Walker Delta constellation with circular orbits.

    Node ID mapping: node_id = plane_idx * sats_per_plane + sat_idx
    """

    def __init__(
        self,
        num_planes: int = 20,
        sats_per_plane: int = 20,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        phase_offset: int = 1,
    ):
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.num_sats = num_planes * sats_per_plane
        self.altitude_km = altitude_km
        self.inclination_rad = np.radians(inclination_deg)
        self.phase_offset = phase_offset

        # Orbital parameters
        self.orbit_radius_m = (R_EARTH_KM + altitude_km) * 1000.0
        self.orbit_radius_km = R_EARTH_KM + altitude_km
        self.orbital_period_s = 2 * np.pi * np.sqrt(
            self.orbit_radius_m ** 3 / MU
        )
        self.orbital_velocity_ms = np.sqrt(MU / self.orbit_radius_m)

        # RAAN spacing between planes
        self.raan_spacing_rad = 2 * np.pi / num_planes
        # True anomaly spacing within a plane
        self.ta_spacing_rad = 2 * np.pi / sats_per_plane
        # Phase offset between adjacent planes
        self.phase_offset_rad = (
            phase_offset * 2 * np.pi / (num_planes * sats_per_plane)
        )

        # Precompute static arrays
        self._plane_ids = np.arange(self.num_sats) // sats_per_plane
        self._sat_ids = np.arange(self.num_sats) % sats_per_plane
        self._raans = self._plane_ids * self.raan_spacing_rad
        self._ta_offsets = (
            self._sat_ids * self.ta_spacing_rad
            + self._plane_ids * self.phase_offset_rad
        )

    def get_positions_eci(self, t_seconds: float) -> np.ndarray:
        """All satellite ECI positions at time t. Returns [N, 3] in km."""
        # True anomaly at time t
        mean_motion = 2 * np.pi / self.orbital_period_s
        ta = self._ta_offsets + mean_motion * t_seconds  # [N]

        # Position in orbital plane
        x_orb = self.orbit_radius_km * np.cos(ta)
        y_orb = self.orbit_radius_km * np.sin(ta)

        # Rotate by inclination and RAAN to ECI
        inc = self.inclination_rad
        cos_raan = np.cos(self._raans)
        sin_raan = np.sin(self._raans)
        cos_inc = np.cos(inc)
        sin_inc = np.sin(inc)

        x_eci = cos_raan * x_orb - sin_raan * cos_inc * y_orb
        y_eci = sin_raan * x_orb + cos_raan * cos_inc * y_orb
        z_eci = sin_inc * y_orb

        return np.stack([x_eci, y_eci, z_eci], axis=-1)

    def get_positions_ecef(self, t_seconds: float) -> np.ndarray:
        """All satellite ECEF positions at time t. Returns [N, 3] in km."""
        eci = self.get_positions_eci(t_seconds)
        theta = OMEGA_EARTH * t_seconds
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_ecef = cos_t * eci[:, 0] + sin_t * eci[:, 1]
        y_ecef = -sin_t * eci[:, 0] + cos_t * eci[:, 1]
        z_ecef = eci[:, 2]
        return np.stack([x_ecef, y_ecef, z_ecef], axis=-1)

    def get_latlon(self, t_seconds: float) -> np.ndarray:
        """Sub-satellite lat/lon at time t. Returns [N, 2] in degrees."""
        ecef = self.get_positions_ecef(t_seconds)
        x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]
        r_xy = np.sqrt(x ** 2 + y ** 2)
        lat = np.degrees(np.arctan2(z, r_xy))
        lon = np.degrees(np.arctan2(y, x))
        return np.stack([lat, lon], axis=-1)

    def get_distances(self, t_seconds: float) -> np.ndarray:
        """Pairwise distance matrix [N, N] in km at time t."""
        pos = self.get_positions_eci(t_seconds)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def get_neighbor_ids(self, node_id: int):
        """Return (intra_fwd, intra_bwd, inter_left, inter_right) node IDs.

        intra-plane neighbors are always valid.
        inter-plane neighbors may be -1 if at constellation edge (wraps around).
        """
        p = node_id // self.sats_per_plane
        s = node_id % self.sats_per_plane

        # Intra-plane: adjacent in same plane (circular)
        intra_fwd = p * self.sats_per_plane + (s + 1) % self.sats_per_plane
        intra_bwd = p * self.sats_per_plane + (s - 1) % self.sats_per_plane

        # Inter-plane: same sat index in adjacent planes (circular)
        left_plane = (p - 1) % self.num_planes
        right_plane = (p + 1) % self.num_planes
        inter_left = left_plane * self.sats_per_plane + s
        inter_right = right_plane * self.sats_per_plane + s

        return intra_fwd, intra_bwd, inter_left, inter_right

    @property
    def epoch_duration_s(self) -> float:
        return 30.0

    @property
    def epochs_per_orbit(self) -> int:
        return int(np.ceil(self.orbital_period_s / self.epoch_duration_s))
