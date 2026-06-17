#!/bin/bash
#SBATCH --job-name=test                 # Nom del job
#SBATCH --output=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/test_output.txt
#SBATCH --error=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/test_error.txt
#SBATCH --time=00:10:00                 # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=1               # Nombre de CPUs per tasca
#SBATCH --mem=4GB                       # Memoria assignada

source /opt/software/miniconda3/bin/activate pinns-repressilator-venv

cd /home/10040984@uvic.local/projects/pinns-repressilator
export PYTHONPATH=/home/10040984@uvic.local/projects/pinns-repressilator:$PYTHONPATH

python -u <<'PY'
import sys, platform
print(f"Python {sys.version}")
print(f"Platform: {platform.node()}")

import numpy as np
import scipy
import matplotlib
import tensorflow as tf
import deepxde as dde
print(f"numpy      {np.__version__}")
print(f"scipy      {scipy.__version__}")
print(f"matplotlib {matplotlib.__version__}")
print(f"tensorflow {tf.__version__}")
print(f"deepxde    {dde.__version__}")

gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs visible: {len(gpus)} — {[g.name for g in gpus] if gpus else 'none (CPU only)'}")

x = tf.constant([1.0, 2.0, 3.0])
print(f"TF test op: sum({x.numpy()}) = {tf.reduce_sum(x).numpy()}")

from scripts.pinns.inverse import run_inverse
from scripts.data.data import generate_dataset
print("Project imports OK")

print("=== Cluster environment check passed ===")
PY
