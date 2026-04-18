"""Top-level exception hierarchy for kvraft.

Subpackages (`cache`, `proxy`, `raft`) extend `KVRaftError` so callers can
catch the whole kvraft surface with one clause when needed, and specific
subclasses when they care about the failure mode.
"""


class KVRaftError(Exception):
    """Base class for all kvraft errors."""
