#!/usr/bin/env python
# encoding: utf-8
"""
Models.py: File containing the enumeration of the models.
"""

from enum import Enum


class Models(Enum):
    """Enumeration of the models types.

    Args:
        Enum (str): enumeration of the models.
    """

    LINEAR = "linear"
    LOGISTIC = "logistic"
