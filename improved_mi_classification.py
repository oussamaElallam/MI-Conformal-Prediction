"""Compatibility entry point for the authoritative PTB-XL pipeline.

Historically this file trained a second independent model, which allowed the same
manuscript architecture to acquire different AUC values across tables. It now
forwards to ``train_split_conformal_model.py`` so every primary result starts from
one saved model and one split manifest.
"""

from train_split_conformal_model import main


if __name__ == "__main__":
    main()
