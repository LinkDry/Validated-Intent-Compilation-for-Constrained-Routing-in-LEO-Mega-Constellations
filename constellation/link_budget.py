"""ISL link budget: FSPL, propagation delay, capacity."""

import numpy as np

# Constants
C_KMS = 299792.458  # speed of light in km/s
C_MS = 2.998e8      # speed of light in m/s


def fspl_db(distance_km: float, frequency_ghz: float = 23.0) -> float:
    """Free-space path loss in dB for ISL (no atmosphere)."""
    d_m = distance_km * 1000.0
    f_hz = frequency_ghz * 1e9
    return 20 * np.log10(4 * np.pi * d_m * f_hz / C_MS)


def propagation_delay_ms(distance_km: float) -> float:
    """One-way propagation delay in milliseconds."""
    return distance_km / C_KMS * 1000.0


def isl_capacity_gbps(
    distance_km: float,
    frequency_ghz: float = 23.0,
    tx_power_dbw: float = 10.0,
    antenna_gain_dbi: float = 35.0,
    bandwidth_ghz: float = 2.0,
    noise_temp_k: float = 290.0,
) -> float:
    """ISL link capacity in Gbps using Shannon bound.

    Simplified V-band RF ISL model. No atmospheric effects (space-to-space).
    """
    # Link budget
    path_loss = fspl_db(distance_km, frequency_ghz)
    # Received power (dBW) = Tx + Tx_gain + Rx_gain - FSPL
    rx_power_dbw = tx_power_dbw + 2 * antenna_gain_dbi - path_loss

    # Noise power
    k_boltzmann = 1.380649e-23  # J/K
    bw_hz = bandwidth_ghz * 1e9
    noise_power_w = k_boltzmann * noise_temp_k * bw_hz
    noise_power_dbw = 10 * np.log10(noise_power_w)

    # SNR
    snr_db = rx_power_dbw - noise_power_dbw
    snr_linear = 10 ** (snr_db / 10)

    # Shannon capacity, capped at practical limit
    capacity_bps = bw_hz * np.log2(1 + snr_linear)
    capacity_gbps = capacity_bps / 1e9

    # Cap at practical modulation limit (~10 Gbps for V-band ISL)
    return float(np.minimum(capacity_gbps, 10.0))
