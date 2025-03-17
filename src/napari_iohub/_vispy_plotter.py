"""
High-performance Vispy-based plotter for napari.

This module provides a Vispy-based plotter for napari that offers high-performance
visualization of large datasets with interactive selection and classification capabilities.
"""

from .vispy_plotter import vispy_plotter_widget

__all__ = ["vispy_plotter_widget"]