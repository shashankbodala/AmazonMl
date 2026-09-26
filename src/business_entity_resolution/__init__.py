"""business_entity_resolution

Top-level package for the Amazon ML Challenge 2026 Business Entity
Resolution project.

Subpackages:
- validation: Leakage-safe validation infrastructure (Task 2)
- normalization: Text normalization for business fields (Task 3A)
- blocking: Baseline blocking with inverted indexes (Task 3A)
"""

__all__ = ["validation", "normalization", "blocking"]
