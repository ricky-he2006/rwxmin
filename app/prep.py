"""Image build time: pickle the processed form of the committed maps (faster to load)."""
import os
import sys

os.environ["BR_PREP"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selector  # noqa: E402
from selcore import Map  # noqa: E402

selector._pkl(selector.MAP, lambda: Map(selector.MAP))
selector._submap()
selector.load_reads()
selector.load_outreads()
selector._replay()
selector._itermap()
selector.load_imports()
try:
    memo_path = os.path.join(HERE, "memo.pkl")
    lite_path = os.path.join(HERE, "memo_lite.pkl")
    if os.path.exists(memo_path) and not os.path.exists(lite_path):
        import pickle
        with open(memo_path, "rb") as fh:
            m = pickle.load(fh)
        lite = {"opt": {k: bool(v) for k, v in m.get("opt", {}).items()}, "rlines": m.get("rlines", {}), "tpch": m.get("tpch", {})}
        with open(lite_path, "wb") as fh:
            pickle.dump(lite, fh, protocol=4)
        print("memo_lite.pkl created")
except Exception as e:
    print("memo_lite creation failed:", repr(e))
try:
    import runner  # noqa: E402
    runner.zygote_import({"tests.test_executor"})
    ok = runner.prebuild_class("tests.test_executor", "TestExecutor", cache=str(selector.PRESET_CACHE))
    print("executor fixtures cached:", ok)
except Exception as e:  # noqa
    print("executor fixtures not cached:", repr(e))
print("maps prepared", sorted(f for f in os.listdir(HERE) if f.endswith(".pkl")))
