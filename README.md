[![Python Versions][python-shield]][python-url]
[![PyPi][pypi-shield]][pypi-url]
[![Tests][tests-shield]][tests-url]
[![License][license-shield]][license-url]
[![Contributors][contributors-shield]][contributors-url]
[![Code Style][codestyle-shield]][codestyle-url]
[![Homepage][homepage-shield]][homepage-url]


# PTE Stats - Python tools for electrophysiology

PTE Stats is an open-source software package for statistics with time series.

It builds upon [PTE](https://github.com/richardkoehler/pte) and provides statistical tools for time-series.
PTE Stats is particularly useful with intracranial EEG (iEEG) data such as local field potentials (LFP) and electrocorticography (ECoG).

## Installing pte-stats

### Stable release

To install the latest stable release, simply type:

```bash
$ pip install pte-stats
```

### Development version

To install the latest delevopment version, first clone this repository:

```bash
$ git clone https://github.com/richardkoehler/pte-stats
```

Then install with the command:

```bash
$ pip install .
```


## Usage

```python
import pte_stats

# Examples coming soon
```

## Contributing

Please feel free to contribute yourselves or to open an **issue** when you encounter a bug or to request a new feature.

For any minor additions or bugfixes, you may simply create a **pull request**. 

For any major changes, make sure to open an **issue** first. When you then create a pull request, be sure to **link the pull request** to the open issue in order to close the issue automatically after merging.

### How to contribute
To contribute yourselves, you should fork this repository, and create your a development branch from your fork.

Then, inside your development branch run the command:

```bash
$ pip install -e .[dev]
```

This will additionally install packages for development, such as black, pylint, mypy and isort.

## License
PTE Stats is licensed under the [MIT license](license-url).

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[python-shield]: https://img.shields.io/static/v1?label=Python&message=3.9|3.10&logoColor=black&labelColor=grey&color=blue
[python-url]: https://pypi.org/project/pte-stats/
[pypi-shield]: https://img.shields.io/static/v1?label=PyPi&message=v0.1.0&logoColor=black&labelColor=grey&color=blue
[pypi-url]: https://pypi.org/project/pte-stats/
[tests-shield]: https://github.com/richardkoehler/pte-stats/actions/workflows/tests.yml/badge.svg
[tests-url]: https://github.com/richardkoehler/pte-stats/actions/workflows/tests.yml
[homepage-shield]: https://img.shields.io/static/v1?label=Homepage&message=ICN&logoColor=black&labelColor=grey&logoWidth=20&color=9cf
[homepage-url]: https://www.icneuromodulation.org/
[contributors-shield]: https://img.shields.io/github/contributors/richardkoehler/pte-stats.svg
[contributors-url]: https://github.com/richardkoehler/pte-stats/graphs/contributors
[license-shield]: https://img.shields.io/static/v1?label=License&message=MIT&logoColor=black&labelColor=grey&logoWidth=20&color=yellow
[license-url]: https://github.com/richardkoehler/pte-stats/blob/main/LICENSE/
[codestyle-shield]: https://img.shields.io/static/v1?label=CodeStyle&message=black&logoColor=black&labelColor=grey&logoWidth=20&color=black
[codestyle-url]: https://github.com/psf/black
