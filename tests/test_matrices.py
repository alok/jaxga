#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from itertools import chain, product
from os import PathLike
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
)

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

@jax.jit
def compute_parity(p):
    return jnp.linalg.det(jax.jacobian(jnp.sort)(p.astype(float))).astype(int)

m1, m2 = jax.random.randint(
    PRNGKey(0), minval=0, maxval=10, shape=(2**10, 2**10)
), jax.random.randint(PRNGKey(2), minval=0, maxval=10, shape=(2**10, 2**10))
# ~8 ms to run



# ~86 ms to run
perm = jax.random.randint(PRNGKey(2), minval=0, maxval=10, shape=[2**10])
