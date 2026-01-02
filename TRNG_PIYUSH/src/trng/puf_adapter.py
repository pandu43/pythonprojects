"""Adapter to use `PUFDesign` as a callable PUF for `PUF_RNG`.

It maps a byte `c_state` to a deterministic challenge index and returns the
PUF response packed into bytes (big-endian bit packing). If the specified
data file is missing, the adapter raises FileNotFoundError.
"""
import sys
import os
import numpy as np

# Ensure TRNG folder is on sys.path so we can import PUFDesign directly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from PUFDesign import PUFDesign


class PUFAdapter:
    """Wraps `PUFDesign` and provides a callable that accepts `c_state` bytes.

    Parameters
    - data_file: path to the RRAM CSV file used by `PUFDesign`.
    - challenge_len: number of bits in the challenge/response (e.g., 32)
    - usecols, skiprows: forwarded to `PUFDesign`.
    """

    def __init__(self, data_file: str, challenge_len: int = 32, usecols=None, skiprows=None):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"PUF data file not found: {data_file}")

        self.challenge_len = challenge_len
        self.puf = PUFDesign(crp_bit=challenge_len, file_path=data_file, usecols=usecols or [1], skiprows=skiprows)

    def __call__(self, c_state: bytes) -> bytes:
        """Return PUF response bytes for given c_state.

        The mapping is deterministic: derive an integer index from `c_state`
        and use it as a PRNG seed to pick the two cell arrays used to compute
        the response. The returned bytes are the packed response bits (big-endian).
        """
        idx = int.from_bytes(c_state, 'big')

        np.random.seed(idx)
        data = self.puf.data
        if self.challenge_len > len(data):
            raise ValueError("challenge_len exceeds available PUF data length")

        a1 = np.random.choice(data, self.challenge_len, replace=False)
        remaining = np.setdiff1d(data, a1)
        a2 = np.random.choice(remaining, self.challenge_len, replace=False)

        response_bits = self.puf.get_response(a1, a2)

        # Pack bits into bytes (big-endian: first bit is MSB of first byte)
        b_len = (len(response_bits) + 7) // 8
        out = bytearray(b_len)
        for i, bit in enumerate(response_bits.tolist()):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            if bit:
                out[byte_idx] |= (1 << bit_idx)

        return bytes(out)
