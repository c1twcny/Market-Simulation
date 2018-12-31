# -----------------------------------------------------------------------------
# Agent Based Market Simulation
#
# Created by: Ta-Wei Chen
# Date: December 28, 2018
# Version: 0.1.0
#
# -----------------------------------------------------------------------------
from agent import Agent
from orderbook import OrderBook

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.text as text

# -----------------------------------------------------------------------------
# Set up global parameters
# -----------------------------------------------------------------------------
AGENTS = 1000 # maximum number of market participants
STYLE = ['large_growth', 'large_blend', 'large_value', \
        'medium_growth', 'medium_blend', 'medium_value', \
        'small_growth', 'small_blend', 'small_value']
RISK = ['aggregsive', 'high', 'medium', 'low', 'safe']
BID = [True, False]
CAPITAL_LOWER = 100.0
CAPITAL_UPPER = 1000.0
CAPITAL_MIN = 0.0 # minimum amount of capital for an agent
SHARES_LOWER = 10.0
SHARES_UPPER = 100.0
SHARES = 100000 # outstanding shares
TIME_STEP = 1
TIME_DAILY = 1000 # maximum number of iteration


# -----------------------------------------------------------------------------
# Set up simulation environment 
# -----------------------------------------------------------------------------
#
# Create agent population
#
# Attribute:
# key: sim_id
# value list:
#   [0] sim_style
#   [1] sim_risk_appetite
#   [2] sim_capital
#   [3] sim_shares
#   [4] sim_bid
#
market_agent_list = []
market_agent_dict = {}
bid_agent_list = []
ask_agent_list = []
realtime_price = []


for index in range(60):
    sim_id = random.randint(1, AGENTS)
    sim_style = random.choice(STYLE)
    sim_risk_appetite = random.choice(RISK)
    sim_capital = round(random.uniform(CAPITAL_LOWER, CAPITAL_UPPER), 2)
    sim_shares = round(random.uniform(SHARES_LOWER, SHARES_UPPER), 2)
    sim_bid = random.choice(BID)
# Create agent population
    market_agent_list.append(Agent(sim_id, sim_style, sim_risk_appetite, \
            sim_capital, sim_shares, sim_bid))
    market_agent_dict[sim_id] = [sim_style, sim_risk_appetite, sim_capital, \
            sim_shares, sim_bid]

# -----------------------------------------------------------------------------
# Define initial values 
# -----------------------------------------------------------------------------
#
# Attributes:
#   bid_agent_list: [] of buy agent ID
#   ask_agent_list: [] of sell agent ID
#
# Dependency
#   market_agent_dict
#
stock_price = 1.0
bid_agent_list = [k for k,v in market_agent_dict.items() if v[4]==True]
ask_agent_list = [k for k,v in market_agent_dict.items() if v[4]==False]

# shuffle the list to simulate the bid/ask order queue
#for i in range(10):
#    random.shuffle(bid_agent_list)
#    random.shuffle(ask_agent_list)


# -----------------------------------------------------------------------------
# Generate order number for bid & ask queues
# -----------------------------------------------------------------------------
#
# Attributes:
#   bid_order_number: {} of bid order number; starting from 'b0'
#   ask_order_number: {} of ask order number; starting from 'a0'
#
# Dependency:
#   OrderBook(): Class
#   order_id: method
#   bid_agent_list: [] of bid agent ID
#   ask_agent_list: [] of ask agent ID
#
bid_order_id = {} 
ask_order_id = {}
bid_order_id, ask_order_id = \
        OrderBook().order_id(bid_agent_list, ask_agent_list)

#print('The ask order ID are:\n', ask_order_id)
#print('The bid order ID are:\n', bid_order_id)

# -----------------------------------------------------------------------------
# Add bid/ask price
# -----------------------------------------------------------------------------
# 
# Attribute
#   tmp_bid: list contains tuple (key, [agent properties list], bid_price)
#   tmp_ask: list contains tuple (key, [agent properties list], ask_price)
#
# Dependency
#   market_agent_dict: main agent properties dictionary
#   bid_agent_list: ID list of all buy agents
#   ask_agent_list: ID list of all sell agents
#
tmp_bid = []
tmp_ask = []


tmp_bid = [(k, v, stock_price+round(random.uniform(-0.5, 0.5), 2)) \
        for k, v in market_agent_dict.items() if k in bid_agent_list]

tmp_ask = [(k, v, stock_price+round(random.uniform(-0.5, 0.5), 2)) \
        for k, v in market_agent_dict.items() if k in ask_agent_list]

# -----------------------------------------------------------------------------
# Diagnostic Block 
# -----------------------------------------------------------------------------
for k, v in market_agent_dict.items():
    if v[4]:
        action = 'Buy'
    else:
        action = 'Sell'
#    print(f"{action}-> AgentID:{k:3d} has ${v[2]:.2f}, {v[0]}, {v[1]} risk tolerance")

#print([(k, v) for k, v in market_agent_dict.items() if v[0]=='small_growth' \
#        and v[1]=='high'])
#print(sum([v[2] for k, v in market_agent_dict.items() if v[0]=='small_growth' \
#        and v[1]=='medium']))

#print(f'Num of Bid Agent {len(bid_agent_list)}: \n{bid_agent_list}')
#print(f'Num of Ask Agent {len(ask_agent_list)}: \n{ask_agent_list}')


# -----------------------------------------------------------------------------
# bid/ask price plot data
# -----------------------------------------------------------------------------
#
# bid_price_plot_data: list of bid_price
# ask_price_plot_data: list of ask_price
#
# Dependency:
#   tmp_bid: list of tuples. 3rd element of the tuple is bid_price
#   tmp_ask: list of tuples. 3rd element of the tuple is ask_price
#
bid_price_plot_data = [p_bid[2]*-1. for p_bid in tmp_bid]
ask_price_plot_data = [p_ask[2] for p_ask in tmp_ask]

# -----------------------------------------------------------------------------
# bid/ask volume
# -----------------------------------------------------------------------------
#
# bid_volume_plot_data: list of bid shares
# ask_volume_plot_data: list of ask shares
#
# Dependency:
#   tmp_bid: list of tuples. 2nd element of the tuple is a [];
#            bid shares is the 4th element in the []
#   tmp_ask: list of tuples. 2nd element of the tuple is a [];
#            ask shares is the 4th element in the []
#

bid_volume_plot_data = [v_bid[1][3]*-1. for v_bid in tmp_bid]
ask_volume_plot_data = [v_ask[1][3] for v_ask in tmp_ask]

# -----------------------------------------------------------------------------
# Order matching & order execution 
# -----------------------------------------------------------------------------
# --- Matching model
# --- Bid: Market price
# --- Ask: Limit price
# --- 
# --- Find the bid agend ID 
for id_bid in range(len(bid_order_id)):
    agent_id = bid_order_id['b'+str(id_bid)] # get bid agent ID
    bid_shares = bid_volume_plot_data[id_bid] * -1.0
    bid_price = bid_price_plot_data[id_bid] * -1.0
# --- Bid market order will take any Ask price,
    tmp_price_shares = 0.0
    tmp_shares = 0.0
    for id_ask in range(len(ask_order_id)):
        ask_shares = ask_volume_plot_data[id_ask]
        ask_price = ask_price_plot_data[id_ask]
        
        if ask_shares <= 0:
            break # No share available; move to the next Ask order
        elif (bid_shares-ask_shares) <= 0:
            # Bid order is totally filled at the Ask price
            # store temp number of shares
            # store temp shares*price info
            # calculate the averaged price per share
            # update the remaining shares at the current Ask index
            # add the averaged price to list for plotting 
            tmp_shares = tmp_shares + bid_shares
            tmp_price_shares = tmp_price_shares + bid_shares*ask_price
            execution_price = tmp_price_shares / tmp_shares
            ask_volume_plot_data[id_ask] = ask_shares - bid_shares
            realtime_price.append(execution_price)
            break # Move to the next id_bid number of the outer loop
        else: # Multiple Ask orders required to fill the Bid order
            # bid_volume_plot_data[id_bid] is the total shares for the order
            tmp_shares = tmp_shares + ask_shares
            tmp_price_shares = tmp_price_shares + ask_shares*ask_price
            bid_shares = bid_shares - ask_shares # new 'bid_shares'; decreasing
            ask_volume_plot_data[id_ask] = 0.0 # remove shares from current id_ask

# --- we will loop through each Ask order,
# --- until the first Bid order is fully filled
# 
for idx in range(len(ask_order_id)):
    agent_id = ask_order_id['a'+str(idx)]
    ask_shares = ask_volume_plot_data[idx]
    ask_price = ask_price_plot_data[idx]
    print(ask_shares, ask_price)


# -----------------------------------------------------------------------------
# Test plots
# -----------------------------------------------------------------------------
plt.close() # Close any plot windwos
#fig, ax = plt.figure()
fig, ax = plt.subplots()
#
# ----------------------- subplot(1, 2, 1) ------------------------------------
#
ax1 = plt.subplot(121)

#plt.hist(bid_price_plot_data, bins=len(tmp_bid), orientation='horizontal')
#plt.hist(ask_price_plot_data, bins=len(tmp_ask), orientation='horizontal')
plt.barh([i for i in range(len(tmp_bid))], bid_price_plot_data, \
        label='Bid', color='red')
plt.barh([i for i in range(len(tmp_ask))], ask_price_plot_data, \
        label='Ask', color='green')
plt.legend(loc='best')
ax1.yaxis_inverted() # invert Y-Axis
#plt.gca().invert_yaxis() # reverse Y-Axis 
#plt.gca().invert_xaxis() # reverse X-Axis
ax1.yaxis.set_label_position('right') # place y-label on the Right
ax1.yaxis.labelpad = 10 # adjust the label spacing -- very handy!!!
plt.ylabel('Order Queue Depth', fontsize=12)
plt.xlabel('Stock Price ($)')
plt.title('Bid/Ask Price', fontsize=15)
#
# ---------------------- subplot(1, 2, 2) -------------------------------------
#
ax2 = plt.subplot(122, sharey=ax1)
plt.barh([i for i in range(len(tmp_bid))], bid_volume_plot_data, \
        label='Bid', color='red')
plt.barh([i for i in range(len(tmp_ask))], ask_volume_plot_data, \
        label='Ask', color='green')
plt.legend(loc='best')
plt.gca().invert_yaxis()
ax2.yaxis.tick_right() # place y-axis tick on the Right
plt.xlabel('Volume (shares)')
plt.title('Bid/Ask Volume', fontsize=15)

plt.show()

# -----------------------------------------------------------------------------
# Submit bid-ask orders 
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Update Order Book 
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Update price-time series plot 
# -----------------------------------------------------------------------------
