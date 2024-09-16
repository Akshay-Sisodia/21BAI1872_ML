import torch
import subprocess
import sys

if torch.cuda.is_available():
    print("GPU is available. Installing GPU requirements.")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements_gpu.txt"]
    )
else:
    print("GPU is not available. Skipping GPU requirements.")
