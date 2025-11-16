#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CONDSAR 训练配置加载和验证脚本
支持从 YAML 配置文件加载训练参数
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: str = None):
        """
        初始化配置加载器

        Args:
            config_path: YAML 配置文件路径
        """
        self.config_path = config_path
        self.config = {}

    def load_yaml(self, path: str) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded config from {path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Config file not found: {path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"❌ Failed to parse YAML: {e}")
            return {}

    def load_json(self, path: str) -> Dict[str, Any]:
        """加载 JSON 配置文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ Loaded config from {path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Config file not found: {path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON: {e}")
            return {}

    def load(self, path: str = None) -> Dict[str, Any]:
        """自动加载配置 (支持 YAML 和 JSON)"""
        path = path or self.config_path

        if not path:
            logger.warning("No config path provided")
            return {}

        if path.endswith('.yaml') or path.endswith('.yml'):
            return self.load_yaml(path)
        elif path.endswith('.json'):
            return self.load_json(path)
        else:
            logger.error(f"Unsupported config format: {path}")
            return {}

    def merge_with_args(self, config: Dict, args: argparse.Namespace) -> Dict:
        """
        将命令行参数合并到配置中 (命令行参数优先级最高)

        Args:
            config: 从配置文件加载的配置
            args: 命令行参数

        Returns:
            合并后的配置
        """
        # 提取命令行参数 (非 None 的值)
        arg_dict = {k: v for k, v in vars(args).items() if v is not None}

        # 浅合并 (命令行参数覆盖配置文件)
        if 'training' in config and 'stage_a' in config['training']:
            training_config = config['training']['stage_a']
            for key, value in arg_dict.items():
                if key in training_config:
                    training_config[key] = value
                    logger.info(f"Override {key}: {value}")

        return config

    def validate(self, config: Dict) -> bool:
        """验证配置的必需字段"""
        required_fields = {
            'data': ['source_dir', 'image_size'],
            'model': ['pretrained_model_name'],
            'training': ['stage_a'],
        }

        for section, fields in required_fields.items():
            if section not in config:
                logger.error(f"❌ Missing section: {section}")
                return False

            for field in fields:
                if field not in config[section]:
                    logger.error(f"❌ Missing field: {section}.{field}")
                    return False

        logger.info("✅ Configuration validated")
        return True

    def print_config(self, config: Dict, section: str = None):
        """打印配置"""
        if section:
            if section in config:
                print(f"\n{'='*60}")
                print(f"[{section.upper()}]")
                print('='*60)
                for key, value in config[section].items():
                    print(f"  {key:30s}: {value}")
        else:
            print(f"\n{'='*60}")
            print("CONFIGURATION")
            print('='*60)
            for section, values in config.items():
                print(f"\n[{section.upper()}]")
                if isinstance(values, dict):
                    for key, value in values.items():
                        print(f"  {key:30s}: {value}")
                else:
                    print(f"  {values}")


def get_training_args_from_config(config: Dict) -> Dict:
    """
    从配置文件提取训练参数

    Returns:
        用于 train.py 的参数字典
    """

    if 'training' not in config or 'stage_a' not in config['training']:
        logger.error("❌ Training config not found")
        return {}

    train_cfg = config['training']['stage_a']

    return {
        'stage': 'a',
        'source_dir': config.get('data', {}).get('source_dir', './data'),
        'batch_size': train_cfg.get('batch_size', 4),
        'num_epochs': train_cfg.get('num_epochs', 100),
        'learning_rate': train_cfg.get('learning_rate', 1e-4),
        'weight_decay': train_cfg.get('weight_decay', 1e-5),
        'device': config.get('device', {}).get('type', 'cuda') + ':' +
                  str(config.get('device', {}).get('device_id', 0)),
        'use_wandb': config.get('wandb', {}).get('enabled', True),
        'output_dir': config.get('output', {}).get('directory', './outputs'),
    }


def generate_training_command(config: Dict, as_string: bool = True) -> str:
    """
    生成训练命令

    Args:
        config: 配置字典
        as_string: 是否返回字符串 (否则返回字典)

    Returns:
        训练命令字符串或参数字典
    """

    args = get_training_args_from_config(config)

    if as_string:
        cmd_parts = ['python scripts/train.py']

        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f'--{key.replace("_", "-")}')
            else:
                cmd_parts.append(f'--{key.replace("_", "-")} {value}')

        return ' '.join(cmd_parts)
    else:
        return args


def main():
    """主函数"""

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='CONDSAR 训练配置加载和生成工具'
    )
    parser.add_argument('--config', type=str, default='config_training.yaml',
                       help='配置文件路径')
    parser.add_argument('--show-config', action='store_true',
                       help='显示配置内容')
    parser.add_argument('--generate-command', action='store_true',
                       help='生成训练命令')
    parser.add_argument('--validate', action='store_true',
                       help='验证配置')
    parser.add_argument('--output-json', type=str, default=None,
                       help='将配置保存为 JSON')

    args = parser.parse_args()

    # 加载配置
    loader = ConfigLoader(args.config)
    config = loader.load(args.config)

    if not config:
        logger.error("Failed to load configuration")
        sys.exit(1)

    # 显示配置
    if args.show_config:
        loader.print_config(config)

    # 验证配置
    if args.validate or args.generate_command:
        if not loader.validate(config):
            logger.error("Configuration validation failed")
            sys.exit(1)

    # 生成命令
    if args.generate_command:
        cmd = generate_training_command(config, as_string=True)
        print("\n" + "="*60)
        print("GENERATED TRAINING COMMAND")
        print("="*60)
        print(cmd)
        print("="*60 + "\n")

    # 保存为 JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Configuration saved to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

