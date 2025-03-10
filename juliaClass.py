import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class JuliaExpense(): 
    def __init__(self, df, month=None, format='%Y%m%d'): 
        self.df = df
        self.df.columns = [i.replace(' ', '') for i in self.df.columns]
        self.df.Date = pd.to_datetime(self.df.Date)
        self.df['Date'] = self.df.Date.dt.strftime(format).str.replace('2035', '2025').apply(pd.to_datetime)
        self.df['date'] = self.df.Date.dt.strftime(format).str.replace('2035', '2025')
        self.df['Type'] = self.df.Type.str.replace('Restaurants', 'Restaurant')
        self.df['Month'] = self.df.Date.dt.strftime('%m')
        self.df['DayWeek'] = self.df.Date.dt.strftime('%a')
        self.df['WeekYear'] = self.df.Date.dt.strftime('%W')
        self.df['dayMonth'] = self.df.Date.dt.strftime('%d')
        self.df['fixedExpense'] = np.where(self.df.Type.isin(['Bills', 'Rent', 'Groceries'])|(self.df.Company.str.contains('TFL')), True, False)
        if month:
            self.df = self.df.loc[self.df.Month.isin(month)]
    def no_fixed(self):
        return self.df.loc[self.df.fixedExpense==False]
    
      
    def tot_monthly_expense(self, grouper=['Month'], tot='Amount', noFixed=True, noBills=False): 
        df = self.no_fixed() if noFixed else self.df
        df = self.no_bills(df) if noBills else df
        #df[tot] = df[tot].astype(float)
        return df.groupby(by=grouper)[tot].sum().reset_index()
    
    def pivot_data(self, grouper=['Month', 'Type'], tot='Amount', indexes = ['Type'], noFixed=True, noBills=False):
        cols = [i for i in grouper if i not in indexes]
        
        return pd.pivot(self.tot_monthly_expense(grouper=grouper, tot=tot, noFixed=noFixed, noBills=noBills), columns=cols, index=indexes, values=tot).fillna(0)
    
    # Show these in the streamlit dashboard
    def pie(self, grouper=['Month', 'Type'], tot='Amount', noFixed=True, byMonth=True):
        fig, ax = plt.subplots(figsize=(5, 15))
        self.pivot_data().plot(kind='pie', subplots=True, ax=ax, autopct='%1.1f%%', startangle=140, labels=None)
        fig.legend(self.pivot_data().index, loc="center left", bbox_to_anchor=(1, 0.5))
        
    def scatter_type(self, grouper=['Month', 'Type'], tot='Amount', indexes = ['Type'], noFixed=True):
        fig, ax = plt.subplots()
        g = sns.scatterplot(data=self.pivot_data(grouper, tot, indexes, noFixed=True), ax=ax)
    
    def daily_stacked(self, grouper=['Type', 'date'], noFixed=True):
        fig, ax = plt.subplots(figsize=(20, 10))
        indexes = grouper[-1]
        if indexes=='DayWeek': 
            self.pivot_data(grouper=grouper, indexes=indexes, noFixed=noFixed).T[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']].T.plot(kind='bar', stacked=True, ax=ax)
        else:
            self.pivot_data(grouper=grouper, indexes=indexes, noFixed=noFixed).plot(kind='bar', stacked=True, ax=ax)

    def top_3_cat(self, type="Gifts", month=None, top=3):
        month_str = f'& Month=="{month}"' if month else ''
        return self.df[self.df.groupby(['Month', 'Type']).Amount.rank(ascending=False)<=top].query(f'(Type=="{type}")'+month_str)[['Month', 'Type', 'Date', 'Company', 'Amount']]
    def get_cummulativeSpend(self, dailyBudget=18.5, noFixed=True, income=2200):
        fig, ax = plt.subplots()
        he = self.pivot_data(grouper=['dayMonth', 'Month'], indexes='Month', noFixed=noFixed).T
        he.index = he.index.astype(int)
        for c in he.columns:
            he[c] = he[c].cumsum()
            he[c].plot(ax=ax)


        he.mean(axis=1).plot(linestyle='--', color='silver', ax=ax, label='avgSpend')
        if noFixed:
            pd.Series([0]+[dailyBudget]*31).cumsum().plot(color='red', linestyle='--', ax=ax, label='Budget')
        else:
            ax.hlines(y=income, xmin=0, xmax=31, color='red', linestyle='--')
        ax.legend()

# Load sample data (replace with actual data loading method)

df = pd.read_csv(r'C:\Users\sbarr\Downloads\Monthly Spending - ALL DATA (1).csv')
df.loc[df.Company=='Popolo','Date'] = '2025-02-02'
julia_expense = JuliaExpense(df)


st.title("Expense Dashboard")

# Create tabs
tabs = st.tabs(["Pie Chart", "Scatter Plot", "Daily Stacked", "Top 3 Categories", "Cumulative Spend"])

with tabs[0]:
    st.header("Pie Chart")
    no_fixed = st.checkbox("Exclude Fixed Expenses", value=True)
    if st.button("Generate Pie Chart"):
        fig = julia_expense.pie(noFixed=no_fixed)
        st.pyplot(fig)

with tabs[1]:
    st.header("Scatter Plot")
    no_fixed = st.checkbox("Exclude Fixed Expenses", value=True)
    if st.button("Generate Scatter Plot"):
        fig = julia_expense.scatter_type(noFixed=no_fixed)
        st.pyplot(fig)

with tabs[2]:
    st.header("Daily Stacked Bar Chart")
    no_fixed = st.checkbox("Exclude Fixed Expenses", value=True)
    if st.button("Generate Stacked Chart"):
        fig = julia_expense.daily_stacked(noFixed=no_fixed)
        st.pyplot(fig)

with tabs[3]:
    st.header("Top 3 Categories")
    category = st.selectbox("Select Category", df["Type"].unique())
    month = st.text_input("Month (optional)")
    top_n = st.slider("Top N", 1, 10, 3)
    if st.button("Show Top 3 Categories"):
        st.dataframe(julia_expense.top_3_cat(type=category, month=month, top=top_n))

with tabs[4]:
    st.header("Cumulative Spend")
    daily_budget = st.number_input("Daily Budget", min_value=0.0, value=18.5)
    income = st.number_input("Income", min_value=0.0, value=2200.0)
    no_fixed = st.checkbox("Exclude Fixed Expenses", value=True)
    if st.button("Show Cumulative Spend"):
        fig = julia_expense.get_cummulativeSpend(dailyBudget=daily_budget, income=income, noFixed=no_fixed)
        st.pyplot(fig)
