"""Rebuild a figure from its saved recipe and say whether it matches the published one.

    python rerun/reproduce.py studies/<study>/figures/<figure>.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lattice.core import runner  # noqa: E402
import lattice.studies.experiments  # noqa: E402,F401


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip())
        return 2
    recipe = Path(sys.argv[1])
    if not recipe.is_file():
        print(f"There is no recipe file at {recipe}. Give the path of a figure's recipe, for example "
              "studies/<study>/figures/<figure>.json")
        return 2
    try:
        json.loads(recipe.read_text())["figure"]
    except (ValueError, KeyError, TypeError):
        print(f"{recipe} is not a figure's recipe. A recipe is the .json file beside a figure.")
        return 2
    try:
        outcome = runner.reproduce_figure(recipe)
    except AttributeError as error:
        print(f"Cannot rebuild this figure yet. {error}")
        return 1
    published = recipe.with_suffix(".svg")
    matches = published.exists() and published.read_text() == outcome["svg"]
    print("The rebuilt figure matches the published one." if matches
          else "The rebuilt figure differs from the published one.")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
