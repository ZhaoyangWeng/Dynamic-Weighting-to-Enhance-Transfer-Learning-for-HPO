# src/test_import.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
print(sys.path) 

import os
print("Current working directory:", os.getcwd())  


blackbox_path = Path(__file__).resolve().parent / "blackbox"
print("Blackbox path:", blackbox_path)
print("Blackbox path exists:", blackbox_path.exists())
print("Blackbox directory contents:", list(blackbox_path.iterdir()))

from blackbox.offline import deepar, fcnet, xgboost, nas102  
print(deepar, fcnet, xgboost, nas102)

