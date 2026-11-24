"""Importing this registers the experiments and figures of every study."""

import importlib

for _number in (1, 2, 3):
    importlib.import_module(f"lattice.studies.study{_number}.experiments")
