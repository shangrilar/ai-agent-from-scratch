"""CH09 snapshot = final version
Changes from CH08:
  - ExecutionContext transfer_to, transfer_tools fields added
This file is identical to the final version of scratch_agent.context.
"""

# CH09 context is identical to the final version, direct re-export
from scratch_agent.context import (  # noqa: F401
    ExecutionContext,
    AgentResult,
    PendingToolCall,
    ToolConfirmation,
)
