# -----------------------------------------------------------------------------
# Order Book
#
# Created by: Ta-Wei Chen
# Date: December 28, 2018
# Version: 0.1.0
#
# -----------------------------------------------------------------------------
#
import math
import random
import numpy as np
import pandas as pd

class OrderBook:
    """Create Order Book

    Attributes:

    """
    bid_price = []
    bid_size = []
    ask_price = []
    ask_size = []

    def __init__(self):
        pass

    def bid(self, bid_price, bid_size):
        self.bid_price = bid_price
        self.bid_size = bid_size

    def ask(self, ask_price, ask_size):
        self.ask_price = ask_price
        self.ask_size = ask_size
