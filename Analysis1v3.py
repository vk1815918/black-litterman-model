# PYTHON 3
# START OF PROGRAM INFORMATION
#-------------------------------------------
# Analysis1.py
    # Author: Sharv Save
    # Purpose:
        # To automatically conduct fundamental analysis of a user inputted stock
    # High level description:
        # To web-scrape financial documents such as Cash Flow Statements,
        # Income Statements, and Balance Sheets
        # to calculate their YOY Revenue Growth, Tangible Asset Value Growth,
        # etc to output either a buy or sell action
# END OF PROGRAM INFORMATION
#-------------------------------------------

import FundamentalAnalysis as fa

class Analysis1():

    def __init__(self, ticker):
        print("\ninside Analysis1.__init__")
        self.ticker = ticker
        self.api_key = '1bd3a664e9e53b21b0264f0a985fce0f'
    # end of function
    
    def perform_analysis1(self):
        print("\ninside Analysis1.perform_analysis1")

        # API Calls
        '''
        if not self.get_data():
            print('error in get_data')
            return False
        # end if'''
        self.df_incomestatement = fa.income_statement(self.ticker, self.api_key)
        self.df_balancesheet = fa.balance_sheet_statement(self.ticker, self.api_key)
        self.df_cashflowstatement = fa.cash_flow_statement(self.ticker, self.api_key)

        # DataFrame Restructuring
        df_incomestatement = self.df_incomestatement.transpose()
        df_balancesheet = self.df_balancesheet.transpose()
        df_cashflowstatement = self.df_cashflowstatement.transpose()

        df_incomestatement = df_incomestatement[:5]
        df_cashflowstatement = df_cashflowstatement[:5]
        df_cashflowstatement = df_cashflowstatement[:5]

        df_incomestatement = df_incomestatement[['calendarYear','netIncome', 'revenue', 'grossProfit']]
        df_balancesheet = df_balancesheet[['calendarYear', 'totalAssets', 'longTermDebt', 'totalCurrentAssets', 'totalCurrentLiabilities', 'commonStock']]
        df_cashflowstatement = df_cashflowstatement[['calendarYear', 'operatingCashFlow']]
        df_fundamental = df_incomestatement.merge(df_balancesheet.merge(df_cashflowstatement, on = 'calendarYear'), on = 'calendarYear')

        self.F_Score_Analysis = {}
        self.F_Score = 0

        #1) ROA
        #Return on Assets:
            # Net Income / Total Assets
        ''' 1 point if positive'''
        Current_Year_ROA = df_fundamental['netIncome'][0] / df_fundamental['totalAssets'][0]
        if(Current_Year_ROA > 0):
            self.F_Score += 1
        else:
            self.F_Score_Analysis["Current Year ROA"] = "The company is not making money off of its investments"
        #2) Change in ROA
            # Current ROA > Last Year ROA
        '''1 point if true'''
        Last_Year_ROA = df_fundamental['netIncome'][1] / df_fundamental['totalAssets'][1]
        if(Current_Year_ROA > Last_Year_ROA):
            self.F_Score += 1
        else:
            self.F_Score_Analysis["Last Year ROA"] = "The company is not making more money off of its investments than last year"
        #3) CFO
            # Operating Cash Flow / Total Assets
        '''1 point if positive'''
        Current_Year_CFO = df_fundamental['operatingCashFlow'][0] / df_fundamental['totalAssets'][0]
        if(Current_Year_CFO > 0):
            self.F_Score += 1
        else:
            self.F_Score_Analysis["Current Year CFO"] = "The company does not have sufficient funds for its operations"
        #4) CFO > ROA
            # CFO > ROA
        '''1 point if true'''
        if(Current_Year_CFO > Current_Year_ROA):
            self.F_Score += 1
        #1) Change in Leverage
            # Long Term Debt / Total Assets
                # Last Year > Current Year
        '''1 point if true'''
        Current_Year_Leverage = df_fundamental['longTermDebt'][0] / df_fundamental['totalAssets'][0]
        Last_Year_Leverage = df_fundamental['longTermDebt'][1] / df_fundamental['totalAssets'][1]
        if(Current_Year_Leverage < Last_Year_Leverage):
            self.F_Score += 1
        else:
            self.F_Score_Analysis["Leverage Change"] = "The company is not gaining more financial leverage"
        #2) Change in Liquidity
            # Current Assets / Current Liabilities
                # Last Year < Current Year
        '''1 point if true'''
        Current_Year_Liquidity = df_fundamental['totalCurrentAssets'][0] / df_fundamental['totalCurrentLiabilities'][0]
        Last_Year_Liquidity = df_fundamental['totalCurrentAssets'][1] / df_fundamental['totalCurrentLiabilities'][1]
        if(Current_Year_Liquidity > Last_Year_Liquidity):
            self.F_Score += 1
        #3) Common Equity
            # Last Year < Current Year
        Current_Year_CE = df_fundamental['commonStock'][0]
        Last_Year_CE = df_fundamental['commonStock'][1]
        if(Current_Year_CE > Last_Year_CE):
            self.F_Score += 1
        #1) Change in Margin
            # Gross Profit / Total Revenue
                # Current Year - Last Year > 0
        '''1 point if true'''
        Current_Year_Margin = df_fundamental['grossProfit'][0] / df_fundamental['revenue'][0]
        Last_Year_Margin = df_fundamental['grossProfit'][1] / df_fundamental['revenue'][1]
        if(Current_Year_Margin - Last_Year_Margin > 0):
            self.F_Score += 1
        #2) Change in Turnover Ratio
            # Total Revenue / Total Assets
                # Current Year - Last Year > 0
        '''1 point if true'''
        Current_Year_Turnover = df_fundamental['revenue'][0] / df_fundamental['totalAssets'][0]
        Last_Year_Turnover = df_fundamental['revenue'][1] / df_fundamental['totalAssets'][1]
        if(Current_Year_Turnover - Last_Year_Turnover > 0):
            self.F_Score += 1
        return True
    # end of function
    
    def getFeedback(self):
        print("\ninside Analysis1.getFeedback")
        # requires def getFeedback() method, to return the dictionary values stored in
        # F_Score_Analysis variable
        return self.F_Score_Analysis, self.F_Score
    # end of function

# end of class