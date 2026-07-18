"""跨实验复用的自定义 GRPO 环境。"""

__all__ = ["QARewardEnv", "QASearchEnv"]


def __getattr__(name):
    """延迟导入，避免本地工具被训练期重依赖阻塞。"""
    if name == "QARewardEnv":
        from common.environments.qa_env import QARewardEnv

        return QARewardEnv
    if name == "QASearchEnv":
        from common.environments.qa_search_env import QASearchEnv

        return QASearchEnv
    raise AttributeError(name)
