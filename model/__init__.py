import os
import warnings

# Prevent joblib/loky Windows subprocess core-detection warning
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

warnings.filterwarnings("ignore", message=".*physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*loky.*")
