import sys
print("Testing debug output to stdout", flush=True)
print("Testing debug output to stderr", file=sys.stderr, flush=True)
