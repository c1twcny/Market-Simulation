# -----------------------------------------------------------------------------
# Agent Based Market Simulation
#
# Created by: Ta-Wei Chen
# Date: December 28, 2018
# Version: 0.1.0
#
# -----------------------------------------------------------------------------
from agent import Agent 

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Set up global parameters
# -----------------------------------------------------------------------------
AGENTS = 1000 # maximum number of market participants
STYLE = ['large_growth', 'large_blend', 'large_value', \
        'medium_growth', 'medium_blend', 'medium_value', \
        'small_growth', 'small_blend', 'small_value']
RISK = ['aggregsive', 'high', 'medium', 'low', 'safe']
BID = [True, False]
CAPITALLOWER = 100.0
CAPITALUPPER = 1000.0
CAPITALMIN = 0.0 # minimum amount of capital for an agent
SHARESLOWER = 10.0
SHARESUPPER = 100.0
SHARES = 100000 # outstanding shares
TIMESTEP = 1

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
for index in range(60):
    sim_id = random.randint(1, AGENTS)
    sim_style = random.choice(STYLE)
    sim_risk_appetite = random.choice(RISK)
    sim_capital = round(random.uniform(CAPITALLOWER, CAPITALUPPER), 2)
    sim_shares = round(random.uniform(SHARESLOWER, SHARESUPPER), 2)
    sim_bid = random.choice(BID)
# Create agent population
    market_agent_list.append(Agent(sim_id, sim_style, sim_risk_appetite, \
            sim_capital, sim_shares, sim_bid))
    market_agent_dict[sim_id] = [sim_style, sim_risk_appetite, sim_capital, \
            sim_shares, sim_bid]

# -----------------------------------------------------------------------------
# Define initial values 
# -----------------------------------------------------------------------------
stock_price = 1.0
bid_agent_list = [k for k,v in market_agent_dict.items() if v[4]==True]
ask_agent_list = [k for k,v in market_agent_dict.items() if v[4]==False]

# shuffle the list to simulate the bid/ask order queue
for i in range(10):
    random.shuffle(bid_agent_list)
    random.shuffle(ask_agent_list)


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
print(len(tmp_bid))
print(len(tmp_ask))

# -----------------------------------------------------------------------------
# Diagnostic Block 
# -----------------------------------------------------------------------------
for k, v in market_agent_dict.items():
    if v[4]:
        action = 'Buy'
    else:
        action = 'Sell'
    print(f"{action}-> AgentID:{k:3d} has ${v[2]:.2f}, {v[0]}, {v[1]} risk tolerance")

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
print(bid_price_plot_data)
print(ask_price_plot_data)

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
print(bid_volume_plot_data)
print(ask_volume_plot_data)

# -----------------------------------------------------------------------------
# Test plots
# -----------------------------------------------------------------------------
fig = plt.figure()
#
# ----------------------- subplot(1, 2, 1) ------------------------------------
#
plt.subplot(1, 2, 1)
#plt.hist(bid_price_plot_data, bins=len(tmp_bid), orientation='horizontal')
#plt.hist(ask_price_plot_data, bins=len(tmp_ask), orientation='horizontal')
plt.barh([i for i in range(len(tmp_bid))], bid_price_plot_data, \
        label='Bid', color='green')
plt.barh([i for i in range(len(tmp_ask))], ask_price_plot_data, \
        label='Ask', color='red')
plt.legend(loc='best')
plt.gca().invert_yaxis() # reverse Y-Axis 
#plt.gca().invert_xaxis() # reverse X-Axis
plt.ylabel('Order queue', )
plt.xlabel('stock price ($)')
plt.title('Bid/Ask Price')
#
# ---------------------- subplot(1, 2, 2) -------------------------------------
#
plt.subplot(1, 2, 2)
plt.barh([i for i in range(len(tmp_bid))], bid_volume_plot_data, \
        label='Bid', color='green')
plt.barh([i for i in range(len(tmp_ask))], ask_volume_plot_data, \
        label='Ask', color='red')
plt.legend(loc='best')
plt.gca().invert_yaxis()
plt.xlabel('shares')
plt.title('Bid/Ask Shares')

plt.show()

# -----------------------------------------------------------------------------
# Submit bid-ask orders 
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Order matching & order execution 
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Update Order Book 
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Update price-time series plot 
# -----------------------------------------------------------------------------
