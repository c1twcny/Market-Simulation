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

    Method:
        order_id

    """
    bid_price = []
    bid_size = []
    ask_price = []
    ask_size = []
    order_num_bid = {}
    order_num_ask = {}

    def __init__(self):
        pass        

    def order_id(self, bid_agent_id, ask_agent_id):
        """ Create ID numbers for bid/ask order
            
            Attributes: bid/ask agent ID []
            Returns: bid/ask order ID {}
        """
        self.bid_agent_id = bid_agent_id
        self.ask_agent_id = ask_agent_id
        
        for idx in range(len(self.bid_agent_id)):
            self.order_num_bid['b'+str(idx)] = self.bid_agent_id[idx]
        for idy in range(len(self.ask_agent_id)):
            self.order_num_ask['a'+str(idy)] = self.ask_agent_id[idy]
        return(self.order_num_bid, self.order_num_ask)


    def new_agent_list(self, new_position):
        """
        Create new bid/ask agent []

        Attribute:
            bid/ask order queues

        Returns:
            new agent list []
            new order volume list []
            new price list []

        """
        self.agent_list = []
        self.volume_list = []
        self.price_list = []

        self.new_pos = new_position
        self.agent_list = [v[0] for v in self.new_pos]
        self.volume_list = [v[1] for v in self.new_pos]
        self.price_list = [v[2] for v in self.new_pos]
        return(self.agent_list, self.volume_list, self.price_list)


    def bid(self, bid_price, bid_size):
        self.bid_price = bid_price
        self.bid_size = bid_size


    def ask(self, ask_price, ask_size):
        self.ask_price = ask_price
        self.ask_size = ask_size
