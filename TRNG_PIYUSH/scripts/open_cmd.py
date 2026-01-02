"""Dry-run copy of `open_cmd.py` into `scripts/`.
"""
import subprocess
import os

def open_cmd_system32():
    """Launch a new Command Prompt window using the cmd.exe located in System32."""
    system32_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "System32", "cmd.exe")
    if not os.path.isfile(system32_path):
        raise FileNotFoundError(f"cmd.exe not found at {system32_path}")
    subprocess.Popen([system32_path], shell=False)

if __name__ == "__main__":
    open_cmd_system32()
