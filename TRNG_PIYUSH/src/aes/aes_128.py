"""AES-128 helper wrapper.

Provides an `AES128` class with ECB and CBC methods using pycryptodome.
Includes a runnable `__main__` that verifies a NIST test vector and a
CBC round-trip.
"""

from typing import Tuple, Optional
import secrets
"""AES-128 helper wrapper.

Provides an `AES128` class with ECB and CBC methods using pycryptodome.
Includes a runnable `__main__` that verifies a NIST test vector and a
CBC round-trip.
"""

from typing import Tuple, Optional
import secrets
import sys


def _require_pycryptodome():
    try:
        from Crypto.Cipher import AES  # noqa: F401
        from Crypto.Util.Padding import pad, unpad  # noqa: F401
    except Exception as e:
        raise ImportError("pycryptodome is required. Install with: pip install pycryptodome") from e


class AES128:
    """Simple wrapper for AES-128 operations using pycryptodome.

    Methods:
        - `encrypt_ecb(block)` / `decrypt_ecb(block)` for single-block ECB
        - `encrypt_cbc(data, iv=None)` returns `(iv, ciphertext)`
        - `decrypt_cbc(iv, ciphertext)` returns plaintext
    """

    BLOCK_SIZE = 16

    def __init__(self, key: bytes):
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError("key must be bytes")
        if len(key) != 16:
            raise ValueError("AES-128 key must be 16 bytes")
        self.key = bytes(key)

    @staticmethod
    def generate_key() -> bytes:
        """Generate a random 16-byte AES-128 key."""
        return secrets.token_bytes(16)

    def encrypt_ecb(self, block: bytes) -> bytes:
        _require_pycryptodome()
        if len(block) != self.BLOCK_SIZE:
            raise ValueError("ECB encrypt expects a single 16-byte block")
        from Crypto.Cipher import AES as _AES
        cipher = _AES.new(self.key, _AES.MODE_ECB)
        return cipher.encrypt(block)

    def decrypt_ecb(self, block: bytes) -> bytes:
        _require_pycryptodome()
        if len(block) != self.BLOCK_SIZE:
            raise ValueError("ECB decrypt expects a single 16-byte block")
        from Crypto.Cipher import AES as _AES
        cipher = _AES.new(self.key, _AES.MODE_ECB)
        return cipher.decrypt(block)

    def encrypt_cbc(self, data: bytes, iv: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Encrypt data in CBC mode. Returns (iv, ciphertext)."""
        _require_pycryptodome()
        from Crypto.Cipher import AES as _AES
        from Crypto.Util.Padding import pad

        if iv is None:
            iv = secrets.token_bytes(self.BLOCK_SIZE)
        if len(iv) != self.BLOCK_SIZE:
            raise ValueError("IV must be 16 bytes")

        cipher = _AES.new(self.key, _AES.MODE_CBC, iv)
        ct = cipher.encrypt(pad(data, self.BLOCK_SIZE))
        return iv, ct

    def decrypt_cbc(self, iv: bytes, ciphertext: bytes) -> bytes:
        _require_pycryptodome()
        from Crypto.Cipher import AES as _AES
        from Crypto.Util.Padding import unpad

        if len(iv) != self.BLOCK_SIZE:
            raise ValueError("IV must be 16 bytes")

        cipher = _AES.new(self.key, _AES.MODE_CBC, iv)
        pt = unpad(cipher.decrypt(ciphertext), self.BLOCK_SIZE)
        return pt


def _run_tests() -> None:
    """Run a couple of quick checks including a known NIST AES-128 vector."""
    print("Running AES-128 tests...")

    # NIST AES-128 ECB example
    key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
    expected_ct = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")

    aes = AES128(key)
    ct = aes.encrypt_ecb(plaintext)
    assert ct == expected_ct, f"ECB encrypt mismatch: {ct.hex()} != {expected_ct.hex()}"
    pt = aes.decrypt_ecb(ct)
    assert pt == plaintext, "ECB decrypt round-trip failed"
    print("ECB test vector: OK")

    # CBC round-trip test
    key2 = AES128.generate_key()
    aes2 = AES128(key2)
    data = b"The quick brown fox jumps over the lazy dog"
    iv, ct2 = aes2.encrypt_cbc(data)
    pt2 = aes2.decrypt_cbc(iv, ct2)
    assert pt2 == data, "CBC round-trip failed"
    print("CBC round-trip: OK")

    print("All tests passed")


def generate_128bit_stream() -> str:
    """Generate a 128-bit bitstream using AES encryption."""
    # Generate random key and plaintext
    key = AES128.generate_key()
    plaintext = secrets.token_bytes(16)  # 16 bytes = 128 bits
    
    # Encrypt to get 128-bit output
    aes = AES128(key)
    ciphertext = aes.encrypt_ecb(plaintext)
    
    # Convert to binary string
    bitstream = ''.join(format(byte, '08b') for byte in ciphertext)
    
    return bitstream


if __name__ == "__main__":
    try:
        _require_pycryptodome()
    except ImportError as e:
        print(e)
        print("Install with: pip install pycryptodome")
        sys.exit(2)

    _run_tests()
    
    # Generate and print 128-bit bitstream
    print("\n" + "="*60)
    print("Generating 128-bit bitstream using AES-128...")
    print("="*60)
    
    bitstream = generate_128bit_stream()
    print(f"\n128-bit Bitstream ({len(bitstream)} bits):")
    print(bitstream)
    print(f"\nAs integer: {int(bitstream, 2)}")
    print(f"As hex: {int(bitstream, 2):032x}")


