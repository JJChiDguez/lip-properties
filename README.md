# README

Proof-of-concept implementation using **sagemath**.

## Requirements:

1. Ensure you have the following files:

  1. `algorithms.py`;
  2. `TestTheorem1.py`;
  3. `TestTheorem2.py`;
  4. `TestLemma2.py`;
  4. `TestLemma3.py`;
  5. `TestRemark7.py`; and
  6. `TestRemark8.py`.

2. **Sagemath** installed

3. **MSolve** library installed


## Description

The scripts require as argument inputs:

- the matrix dimension `n`,
- Groebner basis flag (optional),
- number of samples flag (optional),
- a verbose flag (optional). A single example is excutated in verbose mode.

More precisely,

```bash
Parses command.

options:
  -h, --help            show this help message and exit
  -n DIMENSION, --dimension DIMENSION
                        Matrix dimension: n-by-n matrices
  -gb, --groebner_basis
                        Groebner basis approach
  -v, --verbose         verbose help
  -r NUMBER_OF_SAMPLES, --number_of_samples NUMBER_OF_SAMPLES
                        Number of samples to be used in the Groebner basis approach
```
### Testing ONE random instance

Just run (for example):

```bash
# Linearization approach
% sage -python TestTheorem1.py -n 16 --verbose
% sage -python TestLemma2.py -n 16 --verbose
% sage -python TestLemma3.py -n 16 --verbose
% sage -python TestRemark7.py -n 16 --verbose
# Groebner basis approach
% sage -python TestTheorem1.py -n 16 -gb --verbose
% sage -python TestTheorem2.py -n 16
% sage -python TestLemma2.py -n 16 -gb --verbose
% sage -python TestLemma3.py -n 16 -gb --verbose
% sage -python TestRemark7.py -n 16 -gb --verbose
% sage -python TestRemark8.py -n 16
```

### Testing 25 random instances

Just run (for example):

```bash
# Linearization approach
% sage -python TestTheorem1.py -n 8
% sage -python TestLemma2.py -n 8
% sage -python TestLemma3.py -n 8
% sage -python TestRemark7.py -n 8
# Groebner basis approach
% sage -python TestTheorem2.py -n 8
% sage -python TestLemma2.py -n 8 -gb
% sage -python TestLemma3.py -n 8 -gb
% sage -python TestRemark7.py -n 8 -gb
% sage -python TestRemark8.py -n 8
```


## License

Apache License Version 2.0, January 2004