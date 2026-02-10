import subprocess
import sys

# Run the pipeline and capture output
result = subprocess.run([sys.executable, 'run_scientific_pipeline.py'], 
                       capture_output=True, text=True, encoding='utf-8')

# Print lines containing DEBUG
for line in result.stdout.split('\n'):
    if 'DEBUG' in line:
        print(line)

# Also check for errors
if result.stderr:
    print("ERRORS:")
    for line in result.stderr.split('\n'):
        if 'ERROR' in line or 'Traceback' in line:
            print(line)
