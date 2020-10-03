from . import env

from .utility import *
from .knowknow_base import  *
#from .archive import *


from IPython.core.display import display, HTML, Markdown, Image
from collections import Counter, defaultdict
from random import sample, shuffle
from itertools import chain

from tabulate import tabulate
import re

from pathlib import Path
from os.path import dirname, join
BASEDIR = dirname(__file__)

variable_dir = Path(BASEDIR,"variables")
#variable_dir = Path("C:\\Users\\amcga\\knowknow_variables")

from csv import reader as creader

import networkx as nx
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np