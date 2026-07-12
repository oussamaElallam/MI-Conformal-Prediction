"""Authoritative PTB-XL training and split-conformal artifact generator.

This is the single source of truth for the primary Lightweight CNN metrics.  It
uses PTB-XL folds 1-8 as a development pool, fold 9 for early stopping, and fold
10 as the untouched test set.  The development pool is split into proper
training and calibration with patient/group separation.
"""

from __future