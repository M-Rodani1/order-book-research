# Order book research

Three small simulation studies of a limit order book, worked through one step at a time.

## The questions

1. **Queue position value.** What is an order worth because of its place in the queue?
2. **Square-root impact.** How far does a large order move the price, and what decides the shape of that curve?
3. **Learning liquidity providers.** What changes when the traders who provide liquidity learn, and when does their market break down?

## Where each study stands

- Study 1, Queue position value: not started here yet.
- Study 2, Square-root impact: not started here yet.
- Study 3, Learning liquidity providers: not started here yet.

## How to reproduce a figure

Every figure sits beside a recipe: the configuration and the seeds it was made from. With Python 3.10 or later and NumPy installed, this rebuilds a figure from its recipe and says whether it matches:

```
python rerun/reproduce.py studies/<study>/figures/<figure>.json
```

A figure can only be rebuilt once every part it runs on has been published.

What was done at each step, newest first, is in [LOG.md](LOG.md).

## What I wrote and what was provided

I wrote the study code in each study folder: one file per part, exactly as it passed its checks.

The platform that taught and checked the work, the engine framework and the measurement library were
generated for me. The copy of the engine and measurement code in `rerun/` is that provided code, included
only so that the figures can be rebuilt.

## What these results are

These are small models with chosen parameters. They are not replications of the papers they refer to,
and their numbers are not estimates of anything in a real market. Each figure says what one small model
did under the settings saved in its recipe.
