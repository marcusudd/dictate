import sys

sys.stderr.write(f"ARGS {sys.argv[1:]}\n"); sys.stderr.flush()
if "en" in sys.argv:
    print("ERROR fake failure", flush=True)
    sys.exit(1)
print("READY", flush=True)
for line in sys.stdin:
    sys.stderr.write(f"GOT {line.strip()}\n"); sys.stderr.flush()
