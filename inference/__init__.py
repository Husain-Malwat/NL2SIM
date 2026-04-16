"""
NL2SIM: Natural Language to Simulation Pipeline
A complete system for converting natural language descriptions into MuMax3 scripts.
"""

__version__ = "0.1.0"
__author__ = "NL2SIM Team"

from . import llm
from . import prompts
from . import pipeline
from . import config

__all__ = ["llm", "prompts", "pipeline", "config"]
