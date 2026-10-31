# MIT No Attribution (MIT-0). Burst of real generation requests; exit 0 only if all succeed.
import sys
from concurrent.futures import ThreadPoolExecutor

import llm_solve


def one(i):
    text, usage = llm_solve.chat("http://127.0.0.1:8000/v1", "qwen",
                                 f"Count from 1 to {20 + i} separated by spaces.", 256, 120, 0.6,
                                 {"enable_thinking": False})
    return bool(text) and usage.get("completion_tokens", 0) > 0


try:
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, range(8)))
except Exception as e:  # noqa: BLE001
    print("SMOKE error", repr(e)); sys.exit(1)
print("SMOKE", res); sys.exit(0 if all(res) else 1)
