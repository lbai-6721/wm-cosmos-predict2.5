"""wm4vla — World Model for VLA (skip-dynamics pixel WM on Cosmos-Predict2.5).

This package encapsulates all wm4vla-specific code that extends
NVIDIA Cosmos-Predict2.5 for skip-dynamics world model training:

  Subpackages
  -----------
  datasets   – Kinetix / LIBERO (HDF5 & LeRobot parquet) dataset classes
  configs    – Hydra experiment configs & dataloader registrations
  scripts    – T5 precompute, offline eval, WM visualisation
  doc        – Training docs, project description, data format spec
"""
