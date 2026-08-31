"""LLM-backed fallback for requests the rule-based router (router.py) does
not recognize: a tool registry (tools_schema.py) exposing the same
underlying netlist capabilities router.py dispatches to, and a
provider-agnostic agentic tool-calling client (client.py) driving either the
OpenAI or Anthropic SDK against that registry.
"""
