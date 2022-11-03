## Materials

We modeled all materials with Cauchy's coefficients. Predefined materials could be found in [`../../diffoptics/basics.py`](../../diffoptics/basics.py).

Custom materials could be specified as `<center refractive index at nD> / <Abbe number [um]>`, for example:

```
1.66565/35.64
```

## Format

All surfaces follow the sequence of entries:

```
type   distance   roc   diameter   material
```

For each specific surface, the parameters are pended after the last column:

```
Thorlabs-AC508-1000-A
type distance roc diameter material
O 0      0         100      AIR
S 0      757.9      50.8     N-BK7
S 6.0   -364.7      50.8     N-SF2
S 6.0   -954.2      50.8     AIR
I 996.4   0         50.8     AIR
```
where the symbol means are:

| Symbol | Surface type |      Note      |
| :----: | :----------: | :------------: |
|  `O`   |   Asphere    |  Object plane  |
|  `S`   |   Asphere    |  Lens surface  |
|  `A`   |   Asphere    | Aperture plane |
|  `I`   |   Asphere    |  Image plane   |
