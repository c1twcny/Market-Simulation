# ---------------------------------------------------------------------
# Purpose: Define characteristics of market participating agent
# 
# Created by: Ta-Wei Chen
# Date: December 28, 2018
# Version: 0.1.0
#
# ---------------------------------------------------------------------

class Agent:
    """Define Agent's personality
    
    Attributes:
        agent_id: unique ID for each agent
        trading_style: ['growth', 'value', 'income', 'momentum'] 
        risk_appetite: ['high', 'medium', 'low']
        capital: 100.0 for initial capital
        shares: number of shares owned by an agent
        bid: buyer (True) or seller (False)
    """
    def __init__(self, agent_id=0, trading_style='value', \
            risk_appetite='low', capital=100.0, shares=0.0, bid=True):
        """Initiate Agent with default values"""
        
        self.agent_id = agent_id
        self.trading_style = trading_style
        self.risk_appetite = risk_appetite
        self.capital = capital
        self.shares = shares
        self.bid = True



