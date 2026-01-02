"""
PUF-AES RNG Harness Script

Demonstrates the complete PUF-AES RNG pipeline following Algorithm 1.
Generates multiple 96-bit random bitstreams using:
- PUFDesign for physical unclonable function
- AES-128 for encryption
- PUF_RNG for state management and algorithm implementation
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trng.PUF_RNG import PUF_RNG
from src.trng.puf_adapter import PUFAdapter


def generate_rng_bitstream(num_blocks=10, challenge_len=32, output_dir='Result/RNG', timestamp=None):
    """
    Generate random bitstream using PUF-AES RNG.
    
    Args:
        num_blocks: Number of 96-bit blocks to generate
        challenge_len: Challenge length in bits (default 32)
        output_dir: Directory to save output files
        timestamp: Optional timestamp string
    
    Returns:
        Tuple of (bitstream_string, output_file_path)
    """
    print("="*60)
    print("PUF-AES RNG Bitstream Generator")
    print("="*60)
    
    # Initialize timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    print(f"\n[INFO] Configuration:")
    print(f"  - Timestamp: {timestamp}")
    print(f"  - Number of blocks: {num_blocks}")
    print(f"  - Challenge length: {challenge_len} bits")
    print(f"  - Block size: 96 bits (12 bytes)")
    print(f"  - Total output: {num_blocks * 96} bits ({num_blocks * 12} bytes)")
    
    # Initialize PUF adapter
    print(f"\n[INFO] Initializing PUF adapter...")
    data_file = os.path.join('RRAM_1M_data', '1.9V-1us 1million data.csv')
    
    try:
        adapter = PUFAdapter(data_file, challenge_len=challenge_len, usecols=[1], skiprows=[0])
        print(f"[INFO] PUF adapter initialized with {len(adapter.puf.data)} RRAM data points")
    except FileNotFoundError:
        print(f"[ERROR] RRAM data file not found: {data_file}")
        print(f"[INFO] Falling back to placeholder PUF (SHA-256 based)")
        adapter = None
    except Exception as e:
        print(f"[ERROR] Failed to initialize PUF adapter: {e}")
        print(f"[INFO] Falling back to placeholder PUF (SHA-256 based)")
        adapter = None
    
    # Initialize RNG with 32-bit initial state (4 bytes of zeros)
    initial_state = (0).to_bytes(4, 'big')
    rng = PUF_RNG(c_state=initial_state, counter=0, puf_callable=adapter)
    
    if adapter:
        print(f"[INFO] Using real PUF adapter")
    else:
        print(f"[INFO] Using placeholder PUF (deterministic SHA-256)")
    
    # Generate blocks
    print(f"\n[INFO] Generating {num_blocks} RNG blocks...")
    print("-"*60)
    
    all_rng_bits = []
    all_rng_bytes = []
    
    for i in range(num_blocks):
        r_rng, r_aes = rng.generate_block()
        
        # Convert to binary string
        rng_bits = ''.join(format(b, '08b') for b in r_rng)
        all_rng_bits.append(rng_bits)
        all_rng_bytes.append(r_rng)
        
        print(f"Block {i+1:3d}: {r_rng.hex()} | Counter={rng.counter} | C_state={rng.c_state.hex()}")
    
    # Combine all blocks
    full_bitstream = ''.join(all_rng_bits)
    full_bytes = b''.join(all_rng_bytes)
    
    print("-"*60)
    print(f"[INFO] Generated {len(full_bitstream)} bits total")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binary bitstream
    bin_file = os.path.join(output_dir, f'{timestamp}_RNG_bitstream.txt')
    with open(bin_file, 'w') as f:
        f.write(full_bitstream)
    print(f"[INFO] Saved binary bitstream to: {bin_file}")
    
    # Save hex representation
    hex_file = os.path.join(output_dir, f'{timestamp}_RNG_hex.txt')
    with open(hex_file, 'w') as f:
        f.write(full_bytes.hex())
    print(f"[INFO] Saved hex representation to: {hex_file}")
    
    # Save raw bytes
    raw_file = os.path.join(output_dir, f'{timestamp}_RNG_bytes.bin')
    with open(raw_file, 'wb') as f:
        f.write(full_bytes)
    print(f"[INFO] Saved raw bytes to: {raw_file}")
    
    # Calculate basic statistics
    print(f"\n[INFO] Statistics:")
    ones = full_bitstream.count('1')
    zeros = full_bitstream.count('0')
    ratio = ones / len(full_bitstream) if len(full_bitstream) > 0 else 0
    print(f"  - Total bits: {len(full_bitstream)}")
    print(f"  - Ones: {ones} ({ratio*100:.2f}%)")
    print(f"  - Zeros: {zeros} ({(1-ratio)*100:.2f}%)")
    print(f"  - Balance: {abs(0.5 - ratio)*100:.2f}% deviation from ideal 50/50")
    
    # Save state
    state_file = os.path.join(output_dir, f'{timestamp}_RNG_state.json')
    try:
        rng.save_state(state_file)
        print(f"[INFO] Saved RNG state to: {state_file}")
    except Exception as e:
        print(f"[WARN] Failed to save state: {e}")
    
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    
    return full_bitstream, bin_file


def main():
    parser = argparse.ArgumentParser(
        description='PUF-AES RNG Bitstream Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 blocks (default)
  python puf_aes_rng_harness.py
  
  # Generate 100 blocks with 32-bit challenge
  python puf_aes_rng_harness.py --blocks 100 --challenge-len 32
  
  # Specify custom output directory
  python puf_aes_rng_harness.py --blocks 50 --output Result/MyRNG
        """
    )
    
    parser.add_argument(
        '--blocks', '-b',
        type=int,
        default=10,
        help='Number of 96-bit blocks to generate (default: 10)'
    )
    
    parser.add_argument(
        '--challenge-len', '-c',
        type=int,
        default=32,
        help='Challenge length in bits (default: 32)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='Result/RNG',
        help='Output directory (default: Result/RNG)'
    )
    
    parser.add_argument(
        '--timestamp', '-t',
        type=str,
        default=None,
        help='Custom timestamp (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    try:
        bitstream, output_file = generate_rng_bitstream(
            num_blocks=args.blocks,
            challenge_len=args.challenge_len,
            output_dir=args.output,
            timestamp=args.timestamp
        )
        
        print(f"\n[SUCCESS] RNG bitstream generated successfully!")
        print(f"[SUCCESS] Output file: {output_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate RNG bitstream: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
