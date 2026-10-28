"""MIP baselines for the v5 MT-FJSP.

``model`` is the row-for-row reproduction of Wang et al. (IEEE IoT-J
2025), equations (3)-(8), with no AGVs.  ``model_m2`` is the v5 M2 model
(per-workshop AGV fleets, load/unload stations, empty travel), built as a
fixed-assignment MIP plus an exact AGV chain cover; ``verify_m2`` holds
the independent replays and the fixed-schedule acceptance check;
``run_m2`` runs the 6x6 batch.
"""

from .model import Instance, Solution, load_instance, solve

__all__ = ["Instance", "Solution", "load_instance", "solve"]
