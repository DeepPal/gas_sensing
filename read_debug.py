with open('pipeline_output.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
for line in lines:
    if 'DEBUG' in line:
        print(line.strip())
