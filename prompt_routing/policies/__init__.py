"""Concrete routing/scaffold policy implementations (PHASE4 §8).

Stage 4.3: RuleBasedPromptRouter only. Scaffold policies and SLM stubs arrive
in later stages.
"""

from surrogate_rollout.prompt_routing.policies.rule_based_router import (
    RuleBasedPromptRouter,
)

__all__ = ["RuleBasedPromptRouter"]
