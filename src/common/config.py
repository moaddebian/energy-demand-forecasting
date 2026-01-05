"""
Configuration utilities for parsing config files with environment variable support.
"""
import os
import re
from typing import Union, Optional
from pathlib import Path
import yaml


def parse_config_value(value: Union[str, int, float, None]) -> Union[str, int, float]:
    """
    Parse config value that may contain environment variable placeholders.
    Handles syntax like ${VAR:-default_value}
    
    Args:
        value: Config value (may be string with env var syntax)
    
    Returns:
        Parsed value (string, int, or float)
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float, bool)):
        return value
    
    if not isinstance(value, str):
        return str(value)
    
    # Check if it's an environment variable placeholder
    match = re.match(r'\$\{([^:]+)(?::-([^}]+))?\}', value)
    if match:
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) else None
        env_value = os.getenv(var_name)
        return env_value if env_value is not None else (default_value if default_value is not None else value)
    
    return value


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file. If None, tries to find it in project root.
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try to find config.yaml in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'config' / 'config.yaml'
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config or {}
    
    return {}


def get_config_value(config: dict, key_path: str, default=None, parse_env_vars: bool = True):
    """
    Get a config value by dot-separated key path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'database.host')
        default: Default value if not found
        parse_env_vars: Whether to parse environment variable placeholders
    
    Returns:
        Config value
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
    
    if value is None:
        return default
    
    if parse_env_vars:
        return parse_config_value(value)
    
    return value

