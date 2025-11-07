import yaml
from argparse import ArgumentParser
def load_config():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/base.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"成功加载配置: {args.config}")
    return config
