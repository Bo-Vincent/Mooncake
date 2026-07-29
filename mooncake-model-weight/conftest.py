from pathlib import Path

import mooncake


MODEL_WEIGHT_PACKAGE = str(Path(__file__).parent / "python" / "mooncake")
if MODEL_WEIGHT_PACKAGE not in mooncake.__path__:
    mooncake.__path__.append(MODEL_WEIGHT_PACKAGE)
