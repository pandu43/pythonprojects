"""PUF-based RNG implementing Algorithm 1 (random number generation).

This module provides a testable implementation of the algorithm in the attached
paper figure. It uses a placeholder PUF based on SHA-256 (deterministic) so you
can test and reproduce outputs. Replace `placeholder_puf()` with your real
PUF interface when available.

Behavior summary:
- Internal state: `C_state` (bytes) and `counter` (0..255)
- PUF(C_state) → pseudo-PUF bytes (here SHA256 of C_state)
- Build AES key from padded XOR(C_state, expanded_R_puf_key || counter)
- M_data = PAD(R_PUF, 128-bit)
- R_AES = AES_encrypt(M_data, key) (ECB single-block)
- Update counter and C_state; return R_RNG (12 bytes = 96 bits)

Dependencies: pycryptodome (install with `pip install pycryptodome`).
"""

from typing import Optional, Tuple, Callable
import sys
import os
import json
import hashlib

# Ensure project root is on sys.path so sibling packages can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.aes.aes_128 import AES128


class PUF_RNG:
    """PUF-derived random number generator implementing Algorithm 1.

    Parameters
    - c_state: initial C_state as bytes (defaults to 16 zero bytes)
    - counter: initial counter (0..255)
    - state_file: optional path to persist state (JSON)
    """

    AES_BLOCK = 16  # bytes

    def __init__(self, c_state: Optional[bytes] = None, counter: int = 0, state_file: Optional[str] = None, puf_callable: Optional[Callable[[bytes], bytes]] = None):
        # Use 4-byte (32-bit) C_state by default for strict Algorithm 1 compliance
        default_cstate = (0).to_bytes(4, 'big')
        self.c_state = c_state if c_state is not None else default_cstate
        if len(self.c_state) > self.AES_BLOCK:
            raise ValueError(f"c_state must be at most {self.AES_BLOCK} bytes")
        self.counter = counter & 0xFF
        self.state_file = state_file
        self.puf_callable = puf_callable

    # -------------------- PUF placeholder --------------------
    def placeholder_puf(self, c_state: bytes) -> bytes:
        """Deterministic placeholder for a PUF function.

        Returns SHA-256 digest of the `c_state` and truncates to 16 bytes.
        Replace this method with your real PUF interface which returns bytes.
        """
        h = hashlib.sha256()
        h.update(c_state)
        return h.digest()[: self.AES_BLOCK]

    # -------------------- helpers --------------------
    @staticmethod
    def pad_to_block(data: bytes, block_size: int = AES_BLOCK) -> bytes:
        """Pad or truncate data to `block_size` bytes using simple zero-pad."""
        if len(data) >= block_size:
            return data[:block_size]
        return data + bytes(block_size - len(data))

    @staticmethod
    def xor_bytes(a: bytes, b: bytes) -> bytes:
        return bytes(x ^ y for x, y in zip(a, b))

    def _build_key(self, c_state: bytes, r_puf: bytes, counter: int) -> bytes:
        """Construct a 128-bit AES key according to the algorithm.

        Approach used:
        - Expand `r_puf` by repeating/truncating to match `len(c_state)`
        - Compute x = XOR(c_state, expanded_r_puf)
        - Append single-byte counter to x
        - Pad/truncate result to 16 bytes (AES key)
        """
        # Expand r_puf to match c_state length
        rep = (len(c_state) + len(r_puf) - 1) // len(r_puf)
        expanded = (r_puf * rep)[: len(c_state)]

        x = self.xor_bytes(c_state, expanded)
        x_and_counter = x + bytes([counter & 0xFF])
        key = self.pad_to_block(x_and_counter, self.AES_BLOCK)
        return key

    def _m_data(self, r_puf: bytes) -> bytes:
        """PAD(R_PUF, size=128-bit) — pad/truncate to 16 bytes."""
        return self.pad_to_block(r_puf, self.AES_BLOCK)

    # -------------------- core generate --------------------
    def generate_block(self) -> Tuple[bytes, bytes]:
        """Generate one RNG block following Algorithm 1.

        Returns a tuple (R_rng, R_aes_bytes) where R_rng is the extracted random
        bitstream (12 bytes / 96 bits) and R_aes_bytes is the full AES output
        block (16 bytes) for inspection.
        """
        # 1. R_PUF = PUF(C_state)
        if self.puf_callable is not None:
            r_puf = self.puf_callable(self.c_state)
        else:
            r_puf = self.placeholder_puf(self.c_state)

        # 2-5. Expand R_PUF to size of C_state handled in _build_key

        # 6. key = PAD(C_state XOR R_PUF_key || counter, 128-bit)
        key = self._build_key(self.c_state, r_puf, self.counter)

        # 7. M_data = PAD(R_PUF, size = 128-bit)
        m_data = self._m_data(r_puf)

        # 8. R_AES = AES(M_data, key) using AES helper
        aes = AES128(key)
        r_aes = aes.encrypt_ecb(m_data)

        # 9-13. Update counter
        if self.counter < 255:
            self.counter += 1
        else:
            self.counter = 0

        # 14. C_state = R_AES[1:32] — first 32 bits (4 bytes)
        new_state_prefix = r_aes[:4]
        # Preserve the tail of previous c_state beyond 4 bytes, if present
        if len(self.c_state) > 4:
            self.c_state = new_state_prefix + self.c_state[4:]
        else:
            self.c_state = new_state_prefix

        # 15. R_RNG = R_AES[33:128] — bits 33..128 → bytes 5..16 (12 bytes)
        r_rng = r_aes[4:16]

        # Return RNG and full AES block
        return r_rng, r_aes

    # -------------------- state persistence --------------------
    def save_state(self, path: Optional[str] = None):
        path = path or self.state_file
        if not path:
            raise ValueError("No state_file provided")
        obj = {
            "c_state": self.c_state.hex(),
            "counter": int(self.counter),
        }
        with open(path, "w") as f:
            json.dump(obj, f)

    def load_state(self, path: Optional[str] = None):
        path = path or self.state_file
        if not path:
            raise ValueError("No state_file provided")
        with open(path, "r") as f:
            obj = json.load(f)
        self.c_state = bytes.fromhex(obj["c_state"])[: self.AES_BLOCK]
        self.counter = int(obj["counter"]) & 0xFF


# -------------------- Example / self-test --------------------
if __name__ == "__main__":
    # Try to use PUFAdapter (wraps PUFDesign) if available, otherwise fallback
    rng = None
    try:
        from puf_adapter import PUFAdapter
        data_file = os.path.join('RRAM_1M_data', '1.9V-1us 1million data.csv')
        adapter = PUFAdapter(data_file, challenge_len=32)
        rng = PUF_RNG(c_state=(0).to_bytes(4, 'big'), puf_callable=adapter)
        print('[INFO] Using PUFAdapter as PUF callable')
    except Exception as e:
        print(f'[WARN] PUFAdapter unavailable ({e}). Using placeholder PUF.')
        rng = PUF_RNG(c_state=(0).to_bytes(4, 'big'))

    # generate 5 blocks
    for i in range(5):
        r, full = rng.generate_block()
        print(f"Block {i}: RNG (hex) = {r.hex()} | RNG (bin) = {''.join(format(b, '08b') for b in r)} | AES block = {full.hex()} | AES (bin) = {''.join(format(b, '08b') for b in full)} | counter={rng.counter}")
    # optional: save state
    try:
        rng.save_state(os.path.join(os.getcwd(), "puf_rng_state.json"))
    except Exception:
        pass
