# Code to rerun the figures

Everything in this folder was provided to me, not written by me. It is the minimum of the engine
framework and the measurement library that a figure needs in order to be rebuilt from its recipe.

It contains none of the functions or classes that the studies had me write. Those are only in
`studies/<study>/code/`, so a figure can be rebuilt only once every part it runs on has been published.

- `lattice/core/`: the matching engine framework, the traders' base classes, the experiment runner,
  the measurements and the figure drawing.
- `lattice/studies/`: the experiment and figure definitions of each study.
- `lattice/kit.py`: connects those experiments to the study code I wrote, in `studies/<study>/code/`.
- `reproduce.py`: rebuilds one figure and says whether it matches.

```
python rerun/reproduce.py studies/<study>/figures/<figure>.json
```
