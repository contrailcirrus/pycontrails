"""Humidity scaling methodologies."""

from pycontrails.models.humidity_scaling.humidity_scaling import (
    ConstantHumidityScaling,
    ConstantHumidityScalingParams,
    ExponentialBoostHumidityScaling,
    ExponentialBoostHumidityScalingParams,
    ExponentialBoostLatitudeCorrectionHumidityScaling,
    ExponentialBoostLatitudeCorrectionHumidityScalingParams,
    HistogramMatching,
    HistogramMatchingParams,
    HistogramMatchingWithEckel,
    HistogramMatchingWithEckelParams,
    HumidityScaling,
    HumidityScalingByLevel,
    HumidityScalingByLevelParams,
    eckel_scaling,
    histogram_matching,
    histogram_matching_all_members,
)

__all__ = [
    "ConstantHumidityScaling",
    "ConstantHumidityScalingParams",
    "ExponentialBoostHumidityScaling",
    "ExponentialBoostHumidityScalingParams",
    "ExponentialBoostLatitudeCorrectionHumidityScaling",
    "ExponentialBoostLatitudeCorrectionHumidityScalingParams",
    "HistogramMatching",
    "HistogramMatchingParams",
    "HistogramMatchingWithEckel",
    "HistogramMatchingWithEckelParams",
    "HumidityScaling",
    "HumidityScalingByLevel",
    "HumidityScalingByLevelParams",
    "eckel_scaling",
    "histogram_matching",
    "histogram_matching_all_members",
]
