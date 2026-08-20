# C-MAPSS data provenance

These are the real NASA Turbofan Engine Degradation Simulation data sets
(C-MAPSS), 43 MB across all four sub-datasets:

```
train_FD001..FD004.txt   run-to-failure training units
test_FD001..FD004.txt    right-censored test units
RUL_FD001..FD004.txt     true remaining life at each test unit's last cycle
```

**Original source:** A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage
propagation modeling for aircraft engine run-to-failure simulation",
*International Conference on Prognostics and Health Management*, 2008.
Published by the NASA Prognostics Center of Excellence.

**How these copies were obtained:** downloaded from a public GitHub mirror
(`LahiruJayasinghe/RUL-Net`, `CMAPSSData/`) because NASA's own distribution has
moved repeatedly over the years. The files are byte-identical in format to the
original distribution: whitespace-separated, 26 columns, no header —
`unit, cycle, op_setting_1..3, sensor_1..21`.

**Column count check:** 26 per row in every file; unit counts 100 / 260 / 100 /
249 for train FD001–FD004, matching the published description.

The data is committed rather than downloaded on demand so that `python train.py`
works offline and reproduces the same numbers. If you would rather fetch it
yourself, any mirror of `CMAPSSData/` with the twelve filenames above will do.
