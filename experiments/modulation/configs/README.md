# Configs

## `./sweep-fixed.py`

This script runs a sweep over fixed-moment models (e.g., mean, variance, skewness, kurtosis).

## `./sweep-learned.py`

This script runs a sweep over learned-moment models for raw, central, and standardized moments.

## `./sweep-mixed.py`

This script runs a sweep over mixed-moment models for raw, central, and standardized moments mixed with fixed-moment models. For example, `"mixed-xvector"` will make use of a learned raw moment and a fixed central moment (variance) model producing a `"mixed-xvector"`.
