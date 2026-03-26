"""CH09 snapshot = final version
Changes from CH08:
  - __init__: sub_agents, disallow_transfer_to_peers added
  - run(): transfer_to handling logic
  - _setup_tools(): transfer_tool auto-added
  - _get_transfer_targets(), _find_agent(), _validate_and_set_sub_agents() new
This file is identical to the final version of scratch_agent.agent.
"""

# CH09 agent is identical to the final version, direct re-export
from scratch_agent.agent import Agent  # noqa: F401
