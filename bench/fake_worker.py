import sys

sys.stderr.write(f"ARGS {sys.argv[1:]}\n"); sys.stderr.flush()
print("READY", flush=True)
for line in sys.stdin:
    sys.stderr.write(f"GOT {line.strip()}\n"); sys.stderr.flush()
