from __future__ import annotations

from typing import Literal


def pga_signature(n: int) -> int:
    """0(...+)"""
    return 0 if n == 0 else 1


def sta_signature(n: int) -> int:
    """+(...-)"""
    return 1 if n == 0 else -1


def stap_signature(n: int) -> int:
    """0+(...-)"""
    return 0 if n == 0 else (1 if n == 1 else -1)


def cga_signature(n: int) -> int:
    """+-(...+)"""
    return 1 if n == 0 else (-1 if n == 1 else 1)


def positive_signature() -> Literal[1]:
    """(...+)"""
    return 1


def negative_signature() -> Literal[-1]:
    """(...-)"""
    return -1


def null_signature() -> Literal[0]:
    """(...0)"""
    return 0
