import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np


def empty_lines(n: int) -> None:
    for _ in range(n):
        st.write("")


def percentage_difference(value_1, value_2):
    return round(((value_1 - value_2) / ((value_1 + value_2) / 2)) * 100, 2)


def required_profit_to_break_even(original_amount, amount):
    # Calculate the required profit amount
    profit = original_amount - amount
    # Calculate the percentage profit needed
    percentage_profit_needed = (profit / amount) * 100
    return round(percentage_profit_needed, 2), round(profit)


positional_trades = ["", ""]
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Financial Report", page_icon=":moneybag:", layout="wide")
# Centering the title using CSS
st.markdown(
    """
    <style>
        h1 {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Financial Report")


# Style settings
text_style = """
<style>
    .big-font {
        font-weight: bold;
        font-size: 16px;
        color: #e0e0e0;  /* Light gray color for better visibility on dark background */
    }
    .custom-card {
        background-color: #333333;  /* Darker gray color for card background */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);  /* Optional: Adds a subtle shadow to the card */
    }
    .custom-card ul {
        list-style-type: disc;  /* Defines the type of the list item marker */
        padding-left: 20px;  /* Adjusts the padding to indent the bullet points */
    }
    .custom-card li {
        margin-bottom: 10px;  /* Adds space between list items */
    }
</style>
"""

st.markdown(text_style, unsafe_allow_html=True)

# Basic card-like UI using expander without automatically expanding it.
with st.expander("Rules", expanded=False):
    with st.container():
        st.markdown("""<div class='custom-card'> <p class='big-font'>This is a modernized description inside the 
        card.</p> <ul> <li>Focus on only 10 points</li> <li>Book profits quickly, don't be greedy</li> <li>Have less 
        risk appetite</li> <li>Always prioritize capital preservation. Utilize stop-loss orders</li> <li>Maximum loss 
        for a day is 10 percent of current capital</li> <li>Avoid over trading: Trading too frequently or with 
        excessive volume can lead to increased costs and risks.</li> <li>Aim for a consistent profit target per 
        trade, like 10 points, ensuring it aligns with the average volatility of the instrument being traded.</li> 
        <li>If cumulative losses reach 10% of the capital for the day, halt trading to prevent emotional 
        decisions.</li> <li>Use stop-loss orders for every trade and consistently update Good Till Triggered (GTT) 
        orders to manage and mitigate risks.</li> <li>Avoid the urge to recover all losses in one trade. This often 
        leads to high-risk behaviors and emotional decisions.</li> <li>Only risk 1-2% of your total trading capital 
        per trade to ensure capital preservation.</li>
        </div>
        """, unsafe_allow_html=True)









fl = st.file_uploader(":file_folder: Upload a file", type=(["csv"]))


if fl is not None:
    file_extension = fl.name.split('.')[-1].lower()

    if file_extension == 'csv':
        # Read CSV file directly from the bytes
        df = pd.read_csv(fl, encoding="ISO-8859-1")
        invested_amount = int(st.number_input("Invested Amount"))
        if invested_amount == 0.00:
            invested_amount = 120000

        col1, col2 = st.columns(2)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        # Getting the min and max date
        startDate = pd.to_datetime(df["trade_date"]).min()
        endDate = pd.to_datetime(df["trade_date"]).max()
        with col1:
            date1 = pd.to_datetime(st.date_input("Start Date", startDate))
        with col2:
            date2 = pd.to_datetime(st.date_input("End Date", endDate))

        # df = df[(df["trade_date"] >= date1) & (df["trade_date"] <= date2)].copy()
        data = df.copy()
        group_symbol = data.groupby('symbol')
        INDEX = 0
        ALL_DICT = {}
        for symbol in group_symbol.groups:
            ALL_DICT[INDEX] = {}
            ALL_DICT[INDEX]['SYMBOL'] = symbol
            symbol_df = group_symbol.get_group(symbol)
            group_symbol_transact = symbol_df.groupby('trade_type')

            try:
                symbol_buy_df = group_symbol_transact.get_group('buy')
                buy_quantity = symbol_buy_df['quantity'].sum()
                buy_date = datetime.strptime(str(symbol_buy_df["trade_date"].to_list()[0]), "%Y-%m-%d %H:%M:%S").date()
                buy_sum = (symbol_buy_df['price'] * symbol_buy_df['quantity']).sum()
                buy_avg = np.round(buy_sum / buy_quantity, 3)
                ALL_DICT[INDEX]['buy_qty'] = buy_quantity
                ALL_DICT[INDEX]['buy_date'] = buy_date
                ALL_DICT[INDEX]['buy_avg'] = buy_avg
            except:
                print("error")

            symbol_sell_df = group_symbol_transact.get_group('sell')
            sell_quantity = symbol_sell_df['quantity'].sum()
            sell_date = datetime.strptime(str(symbol_sell_df["trade_date"].to_list()[0]), "%Y-%m-%d %H:%M:%S").date()
            sell_sum = (symbol_sell_df['price'] * symbol_sell_df['quantity']).sum()
            sell_avg = np.round(sell_sum / sell_quantity, 3)

            ALL_DICT[INDEX]['sell_qty'] = sell_quantity
            ALL_DICT[INDEX]['sell_date'] = sell_date

            ALL_DICT[INDEX]['sell_avg'] = sell_avg
            INDEX += 1

        df = pd.DataFrame.from_dict(ALL_DICT, orient='index')
        df["sell_date"] = pd.to_datetime(df["sell_date"])
        #closed Position
        df = df[df['buy_qty'] == df['sell_qty']].copy()
        df = df[(df["sell_date"] >= date1) & (df["sell_date"] <= date2)].copy()

        # df[df['BUY_QUANTITY'] != df['SELL_QUANTITY']] # Open Positions
        # df[df['BUY_QUANTITY'] == df['SELL_QUANTITY']] # Closed Positions

        df['P/L'] = (df['sell_qty'] * df['sell_avg']) - (df['buy_qty'] * df['buy_avg'])
        df['Point_Difference'] = (df['sell_avg']) - (df['buy_avg'])
        df['Percentage_Difference'] = ((df['sell_qty'] * df['sell_avg']) - (df['buy_qty'] * df['buy_avg'])) / (
                df['buy_qty'] * df['buy_avg']) * 100
        df["amount_used"] = df['buy_qty'] * df['buy_avg']
        st.dataframe(df, use_container_width=True)

        total_profit = df['P/L'].sum()
        print(total_profit)
        max_loss = round(df['P/L'].min(), 2)
        max_loss = max_loss if max_loss < 0 else 0
        max_profit = round(df['P/L'].max(), 2)
        # Calculate Wins and Losses
        win_count = (df['P/L'] > 0).sum()
        loss_count = (df['P/L'] <= 0).sum()

        # Daily changes in profit
        daily_profit = df.groupby('buy_date')['P/L'].sum().reset_index()
        cumulative_profit = daily_profit['P/L'].cumsum()  # Calculate cumulative profit
        max_daily_amount_used = df.groupby('buy_date')['amount_used'].max().reset_index()
        color_map = {True: 'green', False: 'red'}
        figure = {
            'data': [
                go.Scatter(
                    x=daily_profit['buy_date'],
                    y=cumulative_profit,  # Use cumulative profit data
                    mode='lines',
                    name='Cumulative Profit',
                    line=dict(color='blue'),  # Color of the line plot
                ),
                go.Scatter(
                    x=daily_profit['buy_date'],
                    y=cumulative_profit,  # Use cumulative profit data
                    mode='markers',
                    name='Cumulative Profit',
                    marker=dict(
                        size=12,  # Increase the size of scatter plot markers
                        color=[color_map[val >= 0] for val in daily_profit['P/L']]
                    ),
                    showlegend=False,  # Remove the legend entry for this trace
                ),
            ],
            'layout': {
                'xaxis': {'title': 'Date'},
                'showlegend': False,
                'title': 'Daily Profit Trend',
                'title_x': 0.5
            },
        }

        figure_1 = px.scatter(
            max_daily_amount_used,
            x='buy_date',
            y='amount_used',

            title='Maximum Amount Used on Each Day',
        ).update_traces(marker=dict(size=12, color='red'), mode='lines+markers').update_layout(title_x=0.5)

        # Example usage:
        current_amount = invested_amount + total_profit
        percentage_needed, required_profit = required_profit_to_break_even(invested_amount, current_amount)
        percentage_needed = 0 if percentage_needed < 0 else percentage_needed
        required_profit = 0 if required_profit < 0 else required_profit

        # Maximum amount used on each day
        max_daily_amount_used = df.groupby('buy_date')['amount_used'].max().reset_index()

        # Add any additional visualizations or analyses here
        empty_lines(1)

        # Styling for the sections
        st.markdown("""
        <style>
            .metric {
                border: 1px solid #FFFFFF; 
                background-color: #1A1A1A;
                padding: 10px;
                border-radius: 5px;
                margin: 5px 0;
                height: 90px;
                text-align: center;
            }
            .header {
                color: #E0E0E0; 
                font-weight: bold;
                font-size: 1.2em;
            }
            .value {
                font-size: 1.5em;
                font-weight: bold;
                color: #E0E0E0;
            }
            .change {
                font-size: 1em;
                color: #FF4500;
            }
            #pushed-content {
                margin-top: 75px;
            }
        </style>
        """, unsafe_allow_html=True)

        # Create a 3-column layout: metrics | spacing | donut chart
        col1, col2, col3 = st.columns([.75, .75, 1])

        # Insert metrics into the left column (col1)
        with col1:
            metrics = {
                "Invested Amount": [invested_amount, ""],
                "Max Loss": ["", max_loss],
                "% Required to BreakEven": [percentage_needed, ""],
            }
            st.markdown('<div id="pushed-content">', unsafe_allow_html=True)
            for key, (value, change) in metrics.items():
                st.markdown(f"""
                    <div class="metric">
                        <div class="header">{key}</div>
                        <div class="value">{value} <span class="change">{change}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            metrics = {
                "Current Balance": [
                    round(current_amount, 2), percentage_difference(current_amount, invested_amount)],
                "Max Profit": [max_profit, ""],
                "Amount Required to BreakEven": [required_profit, ""],
            }
            st.markdown('<div id="pushed-content">', unsafe_allow_html=True)
            for key, (value, change) in metrics.items():
                st.markdown(f"""
                    <div class="metric">
                        <div class="header">{key}</div>
                        <div class="value">{value} <span class="change">{change}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        # Insert donut plot into the right column (col3)
        with col3:
            labels = ['Wins', 'Losses']
            values = [win_count, loss_count]
            colors = ['green', 'red']

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker=dict(colors=colors))])
            st.plotly_chart(fig)
        empty_lines(3)
        st.plotly_chart(figure, use_container_width=True)
        empty_lines(3)
        st.plotly_chart(figure_1, use_container_width=True)

        # Group by week and sum the P/L
        # Convert the sell_date column to datetime format

        df['week'] = df['sell_date'].dt.to_period('W').apply(lambda r: r.start_time)
        grouped = df.groupby('week').sum()['P/L'].reset_index()

        grouped['color'] = ['Profit' if pl >= 0 else 'Loss' for pl in grouped['P/L']]

        fig_1 = px.bar(grouped, x='week', y='P/L',
                       labels={'P/L': 'Profit/Loss', 'week': 'Week'},
                       color='color',
                       color_discrete_map={'Profit': 'green', 'Loss': 'red'},
                       height=500)
        fig_1.update_layout(showlegend=False,
                            title={
                                'text': 'Weekly Aggregated Profit/Loss',
                                'yanchor': 'top',  # anchor the title at its bottom
                                'x': 0.5,  # center the title
                                'xanchor': 'center'  # anchor the title at its center
                            }
                            )

        st.plotly_chart(fig_1, use_container_width=True)

        df['monthly'] = df['sell_date'].dt.to_period('M').apply(lambda r: r.start_time)
        grouped = df.groupby('monthly').sum()['P/L'].reset_index()

        grouped['color'] = ['Profit' if pl >= 0 else 'Loss' for pl in grouped['P/L']]

        fig_2 = px.bar(grouped, x='monthly', y='P/L',
                       labels={'P/L': 'Profit/Loss', 'monthly': 'Month'},
                       color='color',
                       color_discrete_map={'Profit': 'green', 'Loss': 'red'},
                       height=500)
        fig_2.update_layout(showlegend=False,
                            title={
                                'text': 'Monthly Aggregated Profit/Loss',
                                'yanchor': 'top',  # anchor the title at its bottom
                                'x': 0.5,  # center the title
                                'xanchor': 'center'  # anchor the title at its center
                            }
                            )

        st.plotly_chart(fig_2, use_container_width=True)

    else:
        st.error("Please upload a CSV file.")
else:
    st.error("Please upload a file.")
