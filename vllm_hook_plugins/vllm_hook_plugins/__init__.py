import os
import importlib.resources
from pathlib import Path
from typing import Optional

from vllm_hook_plugins.registry import PluginRegistry
from vllm_hook_plugins.hook_llm import HookLLM
from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker
from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker
from vllm_hook_plugins.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from vllm_hook_plugins.analyzers.core_reranker_analyzer import CorerAnalyzer
from vllm_hook_plugins.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer


def get_model_config(config_type: str, model_name: str, config_dir: Optional[str] = None) -> str:
    """Get path to a model config file.
    
    Example:
    config_path = get_model_config('activation_steer', 'ibm-granite/granite-3.1-8b-instruct', config_dir='/my/configs')
    llm = HookLLM(model='...', config_file=config_path, ...)

    Args:
         config_type: str
            Type of config (e.g., 'activation_steer')
        model_name: str
            Model name or path
        config_dir: str
            Optional path to a user-owned config directory. Must mirror the
            bundled layout: <config_dir>/<config_type>/<model_name>.json.
            Takes priority over VLLM_HOOK_CONFIG_DIR and bundled configs.

    Returns:
        Absolute path to the model config JSON file

    """
    model_filename = model_name.split('/')[-1] if '/' in model_name else model_name
    config_file = f'{model_filename}.json'

    if config_dir is not None:
        candidate = Path(config_dir) / config_type / config_file
        if candidate.exists():
            return str(candidate)

    env_dir = os.environ.get('VLLM_HOOK_CONFIG_DIR')
    if env_dir is not None:
        candidate = Path(env_dir) / config_type / config_file
        if candidate.exists():
            return str(candidate)

    if hasattr(importlib.resources, 'files'):
        bundled = importlib.resources.files('vllm_hook_plugins').joinpath(
            f'model_configs/{config_type}/{config_file}'
        )
    else:
        import importlib_resources
        bundled = importlib_resources.files('vllm_hook_plugins').joinpath(
            f'model_configs/{config_type}/{config_file}'
        )
    if Path(str(bundled)).exists():
        return str(bundled)

    raise FileNotFoundError(
        f"No config found for config_type='{config_type}', model='{model_name}'.\n"
        f"To add a custom config, place a JSON file at:\n"
        f"  <your_dir>/{config_type}/{model_filename}.json\n"
        f"then pass config_dir='<your_dir>' or set VLLM_HOOK_CONFIG_DIR='<your_dir>'."
    )


def register_plugins():

    # Register workers
    PluginRegistry.register_worker("probe_hook_qk",       ProbeHookQKWorker,       hooks_on=(True,  False))
    PluginRegistry.register_worker("steer_hook_act",      SteerHookActWorker,      hooks_on=(False, True))
    PluginRegistry.register_worker("probe_hidden_states", ProbeHiddenStatesWorker, hooks_on=(True,  False))

    # Register analyzers
    PluginRegistry.register_analyzer("attn_tracker",   AttntrackerAnalyzer)
    PluginRegistry.register_analyzer("core_reranker",  CorerAnalyzer)
    PluginRegistry.register_analyzer("hidden_states",  HiddenStatesAnalyzer)

__all__ = [
    "PluginRegistry",
    "HookLLM",
    "ProbeHookQKWorker",
    "SteerHookActWorker",
    "ProbeHiddenStatesWorker",
    "AttntrackerAnalyzer",
    "CorerAnalyzer",
    "HiddenStatesAnalyzer",
    "register_plugins",
    "get_model_config",
]