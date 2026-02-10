import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

min_r2 = config['roi']['discovery']['best_sensitivity_min_r2']
print(f'best_sensitivity_min_r2 in config: {min_r2}')
print(f'Type: {type(min_r2)}')
print(f'Is finite: {min_r2 != float("nan") and min_r2 == min_r2}')
