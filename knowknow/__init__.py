from IPython.core.display import display, HTML, Markdown, Image
from collections import Counter, defaultdict
from random import sample, shuffle
from itertools import chain

from tabulate import tabulate
import re

from pathlib import Path
from os.path import dirname, join
import yaml

BASEDIR = dirname(__file__)

from csv import reader as creader

import networkx as nx
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np

from logzero import logger

from collections import OrderedDict

from .exceptions import *

from . import env
from .utility import *

# this uses some imports from here...
from .datastore_cnts import *
from .datastore_cnts.counter import CountHelper
# from .datastore_sql import *

from .code_sharing import *
from . import dataverse
from . import viz

# from .meta_notebook import *
# from .knowknow_base import  *
# from .archive import *

if __name__ == '__main__':
    print("MAINING!")