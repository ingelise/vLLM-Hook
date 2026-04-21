import importlib.resources
from pathlib import Path

from vllm_hook_plugins.registry import PluginRegistry
from vllm_hook_plugins.hook_llm import HookLLM
from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker
from vllm_hook_plugins.workers.steer_activation_worker import SteerHookActWorker
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker
from vllm_hook_plugins.analyzers.attention_tracker_analyzer import AttntrackerAnalyzer
from vllm_hook_plugins.analyzers.core_reranker_analyzer import CorerAnalyzer
from vllm_hook_plugins.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer


def get_model_config(config_type: str, model_name: str) -> str:
    """Get path to a model config file. A convenience function to use included model configurations.

    Args:
        config_type: str
            Type of config (e.g., 'activation_steer')
        model_name: str
            Model name or path

    Returns:
        Absolute path to the model config JSON file

    Example:
        config_path = get_model_config('activation_steer', 'microsoft/Phi-3-mini-4k-instruct')
        llm = HookLLM(model='microsoft/Phi-3-mini-4k-instruct', config_file=config_path, ...)
    """
    model_filename = model_name.split('/')[-1] if '/' in model_name else model_name
    config_file = f'{model_filename}.json'

    if hasattr(importlib.resources, 'files'):
        config_path = importlib.resources.files('vllm_hook_plugins').joinpath(
            f'model_configs/{config_type}/{config_file}'
        )
        return str(config_path)
    else:
        import importlib_resources
        config_path = importlib_resources.files('vllm_hook_plugins').joinpath(
            f'model_configs/{config_type}/{config_file}'
        )
        return str(config_path)


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