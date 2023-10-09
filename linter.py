"""Python script for running static testing"""
import subprocess
import sys

# Get the path of the environment executables from the value of the #
# executed Python binary. Splitting differ for Windows/Linux type paths

if sys.platform == "win32":
    PATH = sys.executable.rsplit("\\", maxsplit=1)[0]
else:
    PATH = sys.executable.rsplit("/", maxsplit=1)[0]

print("[BLACK]", flush=True)
subprocess.run([f"{PATH}/python", "-m", "black", "--check", "."], check=True)


print("[FLAKE 8]", flush=True)
subprocess.run([f"{PATH}/python", "-m", "flake8", "."], check=True)
