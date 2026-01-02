import os
import sys
import subprocess


def main():
	repo_root = os.path.dirname(os.path.abspath(__file__))
	script_path = os.path.join(repo_root, "scripts", "harness_gui_all.py")
	if not os.path.exists(script_path):
		print(f"Error: script not found: {script_path}", file=sys.stderr)
		return 2

	# Prefer project's .venv Python if present, fall back to current interpreter
	venv_dir = os.path.join(repo_root, ".venv")
	if os.name == "nt":
		venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
	else:
		venv_python = os.path.join(venv_dir, "bin", "python")

	if os.path.exists(venv_python):
		python_exe = venv_python
	else:
		python_exe = sys.executable

	print(f"Using Python: {python_exe}")

	cmd = [python_exe, script_path]
	try:
		return subprocess.call(cmd, cwd=repo_root)
	except Exception as e:
		print(f"Failed to run script: {e}", file=sys.stderr)
		return 3


if __name__ == '__main__':
	sys.exit(main())

