"""Create empty package directories so setuptools can resolve the project.

The Dockerfiles install dependencies before copying the source, so that a
change to the code does not invalidate the (slow) dependency layer. But
`pip install .` runs setuptools, and setuptools insists that every directory
named in `[tool.setuptools] packages` actually exists - even though nothing in
them is needed to resolve dependencies.

That list used to be duplicated by hand in each Dockerfile, and it drifted:
`analyst_app` was added to pyproject in phase 9 and the Dockerfiles were never
updated, so every image build failed with

    error: package directory 'analyst_app' does not exist

It was invisible until someone ran `docker compose up` for the first time,
because nothing else reads that list.

Reading the list from pyproject.toml instead means it cannot drift again.
"""

from __future__ import annotations

import pathlib
import sys
import tomllib


def main() -> int:
    config = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    packages = config["tool"]["setuptools"]["packages"]

    for package in packages:
        # "analyst_app.pages" -> analyst_app/pages
        directory = pathlib.Path(*package.split("."))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").touch()

    print(f"stubbed {len(packages)} package directories: {', '.join(packages)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
