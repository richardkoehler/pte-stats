[![Homepage][homepage-shield]][homepage-url]
[![License][license-shield]][license-url]
[![Contributors][contributors-shield]][contributors-url]
[![Code Style][codestyle-shield]][codestyle-url]


# PTE Stats - Python tools for electrophysiology

PTE Stats is an open-source software package for statistics with electrophysiological data

It builds upon [PTE](https://github.com/richardkoehler/pte) and provides statistical tools for time-series, such as intracranial EEG (iEEG) signals such as local field potentials (LFP) electrocorticography (ECoG).

## Installing pte_stats

First, get the current development version of pte_stats using [git](https://git-scm.com/). Type the following command into a terminal:

```bash
git clone https://github.com/richardkoehler/pte_stats
```

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/index.html) to set up a new working environment. To do so navigate to the pte_stats root directory in your terminal and type:

```bash
conda env create -f env.yml
```

This will set up a new conda environment called ``pte_stats``.

To activate the environment then type:

```bash
conda activate pte_stats
```

Finally, to install pte_stats in an editable development version inside your conda enviroment type the following inside the pte_stats root directory:

```bash
conda develop .
```

## Usage

```python
import pte_stats

# Examples
```

## Contributing
Please feel free to contribute yourselves or to open an **issue** when you encounter a bug or would like to add a new feature.

For any minor additions or bugfixes, you may simply create a **pull request**. 

For any major changes, make sure to open an **issue** first. When you then create a pull request, be sure to **link the pull request** to the open issue in order to close the issue automatically after merging.

To contribute yourselves, consider installing the full conda development environment to include such tools as black, pylint and isort:

```bash
conda env create -f env_dev.yml
conda activate pte_stats-dev
```

Continuous Integration (CI) including automated testing are set up.

## License
PTE Stats is licensed under the [MIT license](license-url).

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[homepage-shield]: https://img.shields.io/static/v1?label=Homepage&message=ICN&logoColor=black&labelColor=grey&logoWidth=20&color=9cf&style=for-the-badge
[homepage-url]: https://www.icneuromodulation.org/
[contributors-shield]: https://img.shields.io/github/contributors/richardkoehler/pte_stats.svg?style=for-the-badge
[contributors-url]: https://github.com/richardkoehler/pte_stats/graphs/contributors
[license-shield]: https://img.shields.io/static/v1?label=License&message=MIT&logoColor=black&labelColor=grey&logoWidth=20&color=yellow&style=for-the-badge
[license-url]: https://github.com/richardkoehler/pte_stats/blob/main/LICENSE/
[codestyle-shield]: https://img.shields.io/static/v1?label=CodeStyle&message=black&logoColor=black&labelColor=grey&logoWidth=20&color=black&style=for-the-badge
[codestyle-url]: https://github.com/psf/black
