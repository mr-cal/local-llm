"""Shared primitives used by every command module.

Nothing in here may import from the command modules, so that `llm.core` stays
a leaf of the dependency graph.
"""
