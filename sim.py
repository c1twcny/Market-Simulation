# -----------------------------------------------------------------------------
# Agent Based Market Simulation
#
# Created by: Ta-Wei Chen
# Date: December 28, 2018
# Version: 0.1.0
#
#
# 1/2/2019:
#   add realtime_price = [] in time_step loop
#   add bid/ask_capital_data[]
#
# To-do:
# 1) Define captial requirement before order matching and execution
# 2) Add logics to enable different trading styles and risk tolerance
# 3) Move some codes into Class definition to reduce complexity
# -----------------------------------------------------------------------------
from agent import Agent
from orderbook import OrderBook
from statistics import mean

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
averaged_price = []

for index in range(100):
    sim_id = random.randint(1, AGENTS)
    sim_style = random.choice(STYLE)
    sim_risk_appetite = random.choice(RISK)
    sim_capital = round(random.uniform(CAPITAL_LOWER, CAPITAL_UPPER), 2)
    sim_shares = round(random.uniform(SHARES_LOWER, SHARES_UPPER), 2)
    sim_bid = random.choice(BID)
# Create random agent population
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


tmp_bid = [(k, v, stock_price+round(random.uniform(-0.4, 0.4), 2)) \
        for k, v in market_agent_dict.items() if k in bid_agent_list]

tmp_ask = [(k, v, stock_price+round(random.uniform(-0.4, 0.4), 2)) \
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

#print(ask_price_plot_data)

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
#print(bid_volume_plot_data)

# -----------------------------------------------------------------------------
# bid/ask capital
# -----------------------------------------------------------------------------
bid_capital_plot_data = [c_bid[1][2]*-1. for c_bid in tmp_bid]
ask_capital_plot_data = [c_ask[1][2] for c_ask in tmp_ask]

# -----------------------------------------------------------------------------
# Order matching & order execution 
# -----------------------------------------------------------------------------
# --- Matching model
# --- Bid: Market price
# --- Ask: Limit price
# --- 
# --- Find the bid agend ID
#
# Important!! Make sure you make a copy of an original list. 
# Using [list_new] = [list_old] merely creates a reference pointing to the 
# original list. Any operation done on the new list thus will change the 
# content on the old!
#
# Attribute:
# tmp_ask_volume_data: store updated Ask shares [] after order matching
# tmp_bid_volume_data: store updated Bid shares [] after order matching
#
# Dependency:
#   bid_order_id
#   ask_order_id
#   bid_volume_plot_data
#   ask_volume_plot_data
#   bid_price_plot_data
#   ask_price_plot_data
#
# -----------------------------------------------------------------------------
# Main time stepping loop
# -----------------------------------------------------------------------------
price_queuelength_old = 0 # initialized to 0  
bid_unfilled_sum = 0.0
ask_unfilled_sum = 0.0

for time_stpe in range(10):    
    tmp_bid_volume_data = bid_volume_plot_data.copy()
    tmp_ask_volume_data = ask_volume_plot_data.copy()
    tmp_bid_price_data = bid_price_plot_data.copy()
    tmp_ask_price_data = ask_price_plot_data.copy()
    tmp_bid_capital_data = bid_capital_plot_data.copy() 
    tmp_ask_capital_data = ask_capital_plot_data.copy()

    realtime_price = [] # reset the [] to reduce storage requirement
# ---------------------------------------------------------
# Bid queue loop:
#   Ask queue loop:
#       if-elif-else block: order matching
#
# Bid: market order taks any Ask price
#
    for id_bid in range(len(bid_order_id)):
        agent_id = bid_order_id['b'+str(id_bid)] # get bid agent ID
        bid_shares = bid_volume_plot_data[id_bid] * -1.0
        bid_price = bid_price_plot_data[id_bid] * -1.0
        bid_capital = bid_capital_plot_data[id_bid] * -1.0
        tmp_price_shares = 0.0 # price*shares
        tmp_shares = 0.0

        for id_ask in range(len(ask_order_id)):
            ask_shares = tmp_ask_volume_data[id_ask]
            ask_price = ask_price_plot_data[id_ask]
#        print(id_bid, id_ask, bid_shares, ask_shares)

            if ask_shares <= 0:
                continue # No share available; move on to the next Ask order
            elif (bid_shares-ask_shares) <= 0:
# 1/4/2019
# > add an if-else block to check Bid capital requirement
# >
                if (bid_shares * ask_price > bid_capital):
                    continue # Not enough Bid capital; move on to the next Ask order
                else:
            # Bid order is totally filled at the Ask price
            # 1) Store temp number of shares
            # 2) Store temp shares*price info
            # 3) Calculate the averaged price per share
            # 4) Update the remaining shares at the current Ask index
            # 5) Add the averaged price to list for plotting
            # 6) Exit the Ask-loop and move to the next bid_id in the Bid-loop
                    tmp_shares = tmp_shares + bid_shares # 1)
                    tmp_price_shares = tmp_price_shares + bid_shares*ask_price # 2)
                    execution_price = tmp_price_shares / tmp_shares # 3)
                    tmp_bid_volume_data[id_bid] = 0.0 # bid order at id_bid full filled
                    tmp_ask_volume_data[id_ask] = ask_shares - bid_shares # 4)
                    realtime_price.append(execution_price) # 5)
                    break # 6)
            else: 
            # Multiple Ask orders required to fill a single Bid order
            # bid_volume_plot_data[id_bid] is the total shares for the order
            # The tmp_shares sets to 0 for every new Bid_id. When starting the
            # Ask-loop the initial result should be tmp_shares == ask_shares.
            # Subsequent looping through the Ask-loop will increase the value of
            # tmp_shares, untill the Bid-order is completed filled.
# 1/4/2019
# > add an if-else block to check Bid capital requirement
# >
                if (bid_shares * ask_price > bid_capital):
                    continue
                else:
                    tmp_shares = tmp_shares + ask_shares
                    tmp_price_shares = tmp_price_shares + ask_shares*ask_price
                    bid_shares = bid_shares - ask_shares # new 'bid_shares'; decreasing
                    tmp_ask_volume_data[id_ask] = 0.0 # remove shares from id_ask

# ---------------- End of Bid queue look block ------------

# ---------------------------------------------------------
# Populate new bid/ask volume and price for orders that 
# were executed
# ---------------------------------------------------------
    tmp_bid_pos_active = []
    tmp_bid_pos_close = []
    tmp_ask_pos_active = []
    tmp_ask_pos_close = []
 
    for idx in range(len(bid_order_id)):
        agent_id = bid_order_id['b'+str(idx)] # {agent_id: bid_shares}
        bid_shares = tmp_bid_volume_data[idx] # [shares after order execution]
        bid_price = tmp_bid_price_data[idx]
        if bid_shares == 0.0:
            new_bid_shares = -1.*round(random.uniform(SHARES_LOWER, SHARES_UPPER), 2)
            new_bid_price = -1.*round(mean(realtime_price), 2) + \
                round(random.uniform(-0.2, 0.2), 2)
            tmp_bid_pos_close.append([agent_id, new_bid_shares, new_bid_price])
        else:
            tmp_bid_pos_active.append([agent_id, bid_shares, bid_price])

    for idy in range(len(ask_order_id)):
        agent_id = ask_order_id['a'+str(idy)] # {agent_id: ask_shares]
        ask_shares = tmp_ask_volume_data[idy] # [ask shares after order execution]
        ask_price = tmp_ask_price_data[idy]
        if ask_shares == 0.0:
            new_ask_shares = round(random.uniform(SHARES_LOWER, SHARES_UPPER), 2)
            new_ask_price = round(mean(realtime_price), 2) + \
                round(random.uniform(-0.2, 0.2), 2)
            tmp_ask_pos_close.append([agent_id, new_ask_shares, new_ask_price])
        else:
            tmp_ask_pos_active.append([agent_id, ask_shares, ask_price])

#print(realtime_price)

# ----------------- create averaged price list per time step ---------
    averaged_price.append(mean(realtime_price[-price_queuelength_old:]))
    price_queuelength_new = len(realtime_price)
    price_queuelength_old = price_queuelength_new
    
# ---------------------------------------------------------
# Create new bid/ask order queues, before go back to the
# top of the time stepping loop
# ---------------------------------------------------------
#
# 1/4/2019
# > To-do:
# > Maybe should create a Market-maker to absorve the unfilled bid/ask orders
# > 
    if len(tmp_bid_pos_close) != len(bid_order_id):
        tmp_bid_new_pos = tmp_bid_pos_active + tmp_bid_pos_close
        tmp_ask_new_pos = tmp_ask_pos_close.copy()
        unfilled_bid_sum = round(sum(v[1] * -1.0 for v in tmp_bid_new_pos), 2)
        unfilled_capital = round(unfilled_bid_sum * \
                mean(realtime_price[-price_queuelength_old:]), 2)
        print(f'Unfilled Bid: {unfilled_bid_sum} shares,\t${unfilled_capital}') \
                # tmp_ask_pos_close is fully executed
    elif len(tmp_ask_pos_close) != len(ask_order_id):
        tmp_ask_new_pos = tmp_ask_pos_active + tmp_ask_pos_close
        tmp_bid_new_pos = tmp_bid_pos_close.copy()
        unfilled_ask_sum = round(sum(v[1] for v in tmp_ask_new_pos), 2)
        unfilled_capital = round(unfilled_ask_sum * \
                mean(realtime_price[-price_queuelength_old:]), 2)
        print(f'Unfilled Ask: {unfilled_ask_sum} shares,\t${unfilled_capital}') \
                #tmp_bid_pos_close is fuly executed
        
    else:
        print('Complete fill for both Bid & Ask')
        tmp_full_list = tmp_bid_pos_close + tmp_ask_pos_close
        print(tmp_full_list)
        random.shuffle(tmp_full_list)

    bid_agent_list, bid_volume_plot_data, bid_price_plot_data = \
            OrderBook().new_agent_list(tmp_bid_new_pos)
    ask_agent_list, ask_volume_plot_data, ask_price_plot_data = \
            OrderBook().new_agent_list(tmp_ask_new_pos)

    bid_order_id = {} 
    ask_order_id = {}
    bid_order_id, ask_order_id = \
        OrderBook().order_id(bid_agent_list, ask_agent_list)

# ----------------------- End of time_step loop --------------------------------

# --- create new bid/ask shares

# ---u create new bid/ask price

# Advance Time Step, back to the loop



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

# -------------------------------------
# Figure 2
# Realtime price
# -------------------------------------
plt.figure(2)
plt.plot(realtime_price)


# -------------------------------------
# Figure 3 
# Bid/Ask volume after order matching
# -------------------------------------
plt.figure(3)
ax3 = plt.subplot(111)

plt.barh([i for i in range(len(tmp_bid))], tmp_bid_volume_data, \
        label='Bid', color='red')
plt.barh([i for i in range(len(tmp_ask))], tmp_ask_volume_data, \
        label='Ask', color='green')
plt.legend(loc='best')
#ax3.yaxis_inverted() # invert Y-Axis
plt.gca().invert_yaxis() # reverse Y-Axis 
#plt.gca().invert_xaxis() # reverse X-Axis
ax3.yaxis.set_label_position('left') # place y-label on the Right
ax3.yaxis.labelpad = 10 # adjust the label spacing -- very handy!!!
plt.ylabel('Order Queue Depth', fontsize=12)
plt.xlabel('Volume (shares)')
plt.title('Bid/Ask Volume after Order Matching', fontsize=15)


# -------------------------------------
# Figure 4 
# Plot averaged price time series
# -------------------------------------
plt.figure(4)
plt.plot(averaged_price)




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
