# DARIA (star formation rate Density with bAlmeR alpha Intensity mApping)

This is a lines-only near-infrared EBL model originally intended to constrain the cosmic star formation history through $H\alpha$ intensity mapping and galaxy-EBL cross-correlations, although it can be applied more generally.

## Installation

DARIA has a few non-standard dependencies:

1. [hmf](https://hmf.readthedocs.io/en/latest/index.html)

2. [dust_attenuation](https://dust-attenuation.readthedocs.io/en/latest/)

Once you have those installed and are in this top-level directory of the repo, run:

``` pip install .```

## File configuration

`daria` will save and retrieve data products that are computationally expensive to produce. To make sure `daria` knows where to put these data products, perform the following steps:

1. In whichever shell configuration file you use (e.g., `.bashrc`), add the line below (customized to your preferred (existing) directory) and source the file to apply the changes.

``` export DARIA="/absolute/path/to/where/you/want/to/store/data/products" ```

2. Copy the config template from `daria` to the directory you defined above. From the current directory, you can run:

``` cp src/daria/bookkeeping/config_template.yaml $DARIA/config.yaml ```

3. Assuming you have successfully installed `daria`, run:

```
python
from daria.bookkeeping import Config
c = Config()
exit()
```