#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 21:44:52 2025

Set up packages for running cellpose
@author: sammxie
"""
# utils/env.py
import sys
import subprocess
import importlib
from typing import Optional

def require_package(package: str, min_version: Optional[str] = None):
    """
    Returns
    -------
    module : the imported module object
    """
    try:
        mod = importlib.import_module(package)
        print(f"{package} already installed.")
    except ImportError:
        print(f" {package} not found. Installing with pip...")
        cmd = [sys.executable, "-m", "pip", "install", package]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to install '{package}'. Tried command: {' '.join(cmd)}"
            ) from e
        mod = importlib.import_module(package)
        print(f"{package} installed successfully.")

    if min_version:
        try:
            from importlib.metadata import version
            from packaging.version import Version
            installed = version(package)
            if Version(installed) < Version(min_version):
                raise RuntimeError(
                    f"{package}>={min_version} required, found {installed}.\n"
                    f"Upgrade with: pip install -U {package}"
                )
        except Exception:
            # If version check fails, just skip
            pass

    return mod

