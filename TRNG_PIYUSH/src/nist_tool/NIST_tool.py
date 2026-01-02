import subprocess
import os
import sys


def prompt_with_default(prompt_text, default, cast=str):
    """
    Ask user for input.
    If empty, return default.
    """
    user_input = input(f"{prompt_text} [{default}]: ").strip()
    if user_input == "":
        return default
    try:
        return cast(user_input)
    except ValueError:
        print(f"[WARN] Invalid input, using default: {default}")
        return default


def run_assess(
    num_bits: int = 5000,
    input_file: str = "./data/data.pi",
    num_tests: int = 10
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sts_dir = os.path.join(script_dir, "sts-2.1.2")
    assess_path = os.path.join(sts_dir, "assess.exe")

    if not os.path.exists(assess_path):
        raise FileNotFoundError(f"assess.exe not found: {assess_path}")

    # Interactive inputs to assess.exe
    user_inputs = "\n".join([
        "0",
        input_file,
        "1",
        "0",
        str(num_tests),
        "0"
    ]) + "\n"

    print("\n[INFO] Running NIST STS with:")
    print(f"  num_bits   = {num_bits}")
    print(f"  input_file = {input_file}")
    print(f"  num_tests  = {num_tests}\n")

    subprocess.run(
        [assess_path, str(num_bits)],
        cwd=sts_dir,
        input=user_inputs,
        text=True,
        shell=True
    )


# ==================================================
# Script Entry Point
# ==================================================
if __name__ == "__main__":

    # Step 1: CLI arguments (highest priority)
    if len(sys.argv) > 1:
        try:
            num_bits = int(sys.argv[1])
        except ValueError:
            num_bits = 5000
    else:
        num_bits = None

    input_file = sys.argv[2] if len(sys.argv) > 2 else None
    num_tests = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Step 2: Interactive prompt (fallback)
    if num_bits is None:
        num_bits = prompt_with_default("Enter number of bits", 5000, int)

    if input_file is None:
        input_file = prompt_with_default(
            "Enter input file path (relative to sts dir)",
            "./data/data.pi",
            str
        )

    if num_tests is None:
        num_tests = prompt_with_default("Enter number of tests", 10, int)

    # Step 3: Run
    run_assess(num_bits, input_file, num_tests)
