# PYTHON 3
# START OF PROGRAM INFORMATION
#-------------------------------------------
# Analysis2v3.py
    # Author: Sharv Save
    # Purpose:
        # To conduct public sentiment analysis of an equity through the bag-of-words model
    # High level description:
        # To web-scrape 4 different articles that relate to the earnings of a certain stock,
        # and then to analyze and look for certain key-words that indicate public sentiment
        # of the stock, ie. "overbought" "oversold" "meets expectations" etc.
# END OF PROGRAM INFORMATION
#-------------------------------------------

import os
import sys
import pandas as pd
import numpy as np
import requests as rq
import matplotlib as plt

class Analysis2():
    
    def __init__(self, ticker):
        print("\ninside Analysis2.__init__")
        self.ticker = ticker
    # end of function
    
    def perform_analysis2(self):
        print("\ninside Analysis2.perform_analysis2")
        return True
    # end of function

# end of class