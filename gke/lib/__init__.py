"""Library helpers extracted from gke/deploy.py.

Step 2 of the rebuild plan in gke/PLAN.md splits the original 678-line
deploy.py into focused modules under this package. Imports from these
modules are stable; the gke/deploy.py CLI is being deprecated and will
be replaced by gke/orchestrator.py in step 6.
"""
