## GUV Fluorescence Analyzer

**A Python tool for quantitative analysis of fluorescence time-series data in GUV experiments**

## Overview

This project implements an analysis workflow for fluorescence time-series measurements, with an initial focus on fitting exponential-rise responses.
The tool estimates model parameters, evaluates fit quality, and generates diagnostic plots to support quantitative analysis of biophysical experiments.

## Features
- CSV data loading and validation
- Exponential-rise curve fitting
- Estimation of baseline fluorescence, amplitude, onset time, and time constant
- Goodness-of-fit metrics and residual analysis
- Export of diagnostic plots
- Automated tests for the mathematical model

Mathematical model
For times before the fitted onset, the signal is modeled as a constant baseline. After onset, the fluorescence response follows:

[
F(t)=F_0+A(1-e^{-(t-t_0)/tau})
]


where:
- \(F_0\): baseline fluorescence intensity
- \(A\): amplitude of the fluorescence change
- \(t_0\): fitted onset time of the modeled response
- \(tau\): time constant of the fitted fluorescence response

## Input format

The input CSV file must contain two columns:

| Column | Description | Unit |
|---|---|---|
| `time_s` | Measurement time | s |
| `fluorescence` | Measured fluorescence intensity | Instrument-dependent |

## Example

A synthetic example dataset is provided at `data/example_synthetic.csv`.

The dataset is intended for testing the software workflow. It is not experimental data.

## Scientific limitations

The fitted time constant describes the response of the measured fluorescence signal under the chosen model. It does not automatically represent the time constant of water transport across a membrane.

Estimating a water-permeability coefficient (\(P_f\)) requires a validated physical relationship between fluorescence, vesicle volume or dye concentration, and osmotic water flux. This calculation is not implemented in the current version.

Fit quality should be assessed using residuals and model assumptions, not \(R^2\) alone.

## Project status

Early-stage research software. The initial implementation focuses on transparent model fitting, data validation, and diagnostic visualization.

## Planned improvements

- Parameter uncertainty estimation
- More comprehensive automated tests
- Comparison of alternative kinetic models
- Support for experimental metadata
- Validation against independently characterized datasets

## Technologies

Python · NumPy · pandas · SciPy · Matplotlib · pytest
