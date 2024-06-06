# napari-iohub

[![License BSD-3](https://img.shields.io/pypi/l/napari-iohub.svg?color=green)](https://github.com/czbiohub/napari-iohub/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-iohub.svg?color=green)](https://pypi.org/project/napari-iohub)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-iohub.svg?color=green)](https://python.org)
[![tests](https://github.com/czbiohub/napari-iohub/workflows/tests/badge.svg)](https://github.com/czbiohub/napari-iohub/actions)
[![codecov](https://codecov.io/gh/czbiohub/napari-iohub/branch/main/graph/badge.svg)](https://codecov.io/gh/czbiohub/napari-iohub)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-iohub)](https://napari-hub.org/plugins/napari-iohub)

OME-Zarr viewer plugin for napari with iohub as the I/O backend.

Widgets are provided for different visualization use cases:

- Multi-well plate (high-content screening, HCS) viewer
- Label editor

Usage documentation can be found in the [wiki](https://github.com/czbiohub-sf/napari-iohub/wiki).

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## Installation

First, install [napari](https://napari.org/stable/tutorials/fundamentals/installation.html#installation).

You can then install `napari-iohub` via [pip]:

```sh
git clone https://github.com/czbiohub/napari-iohub.git
pip install ./napari-iohub
```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-iohub" is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

---
Designed for the intracellular dashboard project
by the Computational Microscopy team at the Chan Zuckerberg Biohub SF.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[file an issue]: https://github.com/czbiohub/napari-iohub/issues
