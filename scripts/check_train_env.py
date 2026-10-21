"""Which training libraries are available in this environment."""
import importlib.metadata as md
import importlib.util as u

for m in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "trl"]:
    found = u.find_spec(m) is not None
    print(f"{m:14s} {'installed ' + md.version(m) if found else 'MISSING'}")
