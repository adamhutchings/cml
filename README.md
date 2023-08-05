# C ML

## Purpose

This project is an implementation of a classical neural network in plain C, with
no dependencies other than the C standard library.

## Requirements

- `gcc`

## Build Instructions

Use `./scripts/dbuild.sh` and `./scripts/build/drun.sh` to run the debug version
and `./scripts/rbuild.sh` and `./scripts/rrun.sh` for the release version.

## Current Status

Machine learning is working well, but may need some more optimizations to get a
good neural network.

## Capabilities

- Can train on IRIS in 0.02 seconds. (Run the `successcounter.py` script to test
this yourself ONLY IF `main.c` is set to run `iristest` instead of `mnisttest`.)
- Can recognize handwritten digits with 75% accuracy in under 90 seconds.

## Strategies Used

- Backpropagation -- a hand-derived and hand-coded implementation of partial
derivatives in a neural network.
- Learning rate adjustment -- adjusts learning rate based on the error of going
a single step down in the gradient.
- Repeated descent -- calculates a gradient once and goes in that direction
repeatedly to avoid expensive recalculation.
- Multithreading -- can multithread loss and gradient calculations for
time-efficient calculation.
