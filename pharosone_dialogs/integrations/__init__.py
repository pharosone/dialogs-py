"""Framework integrations (LangChain, LiteLLM).

Import the submodule you need::

    from pharosone_dialogs.integrations.langchain import PharosCallbackHandler
    from pharosone_dialogs.integrations.litellm import pharos_litellm_callback

Neither framework is a dependency of this package: ``langchain-core`` is only
required when a handler is actually instantiated, and ``litellm`` is never
imported at all (the callback is plain duck-typing).
"""
