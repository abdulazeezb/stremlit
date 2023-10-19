import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import utills as util
import css_styles as styles
import datetime as datetime
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

st.set_page_config(layout="wide")

positional_trades = ["DABUR23SEP590CE", "ABCAPITAL23OCT195CE", "ABCAPITAL23SEP220CE", "ADANIENT23OCT2550CE",
                     "ATUL23OCT7200CE", "HDFCLIFE23OCT600PE", "GUJGASLTD23OCT440CE", "INDIAMART23OCT3000CE",
                     "LICHSGFIN23OCT500CE", "SIEMENS23OCT3700CE", "SAIL23OCT93CE", "NMDC23OCT170CE",
                     "APOLLOTYRE23OCT400CE", "ZEEL23OCT315CE"]


def main_page():
    if not util.check_password():
        st.stop()

    st.markdown(styles.text_style, unsafe_allow_html=True)

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

    def combine_intraday(data):
        group_symbol = data.groupby('symboldate')
        index = 0
        all_dict = {}
        for symbol in group_symbol.groups:
            all_dict[index] = {}
            all_dict[index]['SYMBOL'] = symbol
            symbol_df = group_symbol.get_group(symbol)
            group_symbol_transact = symbol_df.groupby('trade_type')
            try:
                symbol_buy_df = group_symbol_transact.get_group('buy')
                buy_quantity = symbol_buy_df['quantity'].sum()
                buy_date = datetime.datetime.strptime(str(symbol_buy_df["trade_date"].to_list()[0]),
                                                      "%Y-%m-%d %H:%M:%S").date()
                buy_sum = (symbol_buy_df['price'] * symbol_buy_df['quantity']).sum()
                buy_avg = np.round(buy_sum / buy_quantity, 3)
                all_dict[index]['SYMBOL'] = symbol_buy_df['symbol'].to_list()[0]
                all_dict[index]['buy_qty'] = buy_quantity
                all_dict[index]['buy_date'] = buy_date
                all_dict[index]['buy_avg'] = buy_avg
            except:
                print("error")
            try:
                symbol_sell_df = group_symbol_transact.get_group('sell')
                sell_quantity = symbol_sell_df['quantity'].sum()
                sell_date = datetime.datetime.strptime(str(symbol_sell_df["trade_date"].to_list()[0]),
                                                       "%Y-%m-%d %H:%M:%S").date()
                sell_sum = (symbol_sell_df['price'] * symbol_sell_df['quantity']).sum()
                sell_avg = np.round(sell_sum / sell_quantity, 3)
            except:
                sell_quantity = 0
                sell_date = "1995-10-03"
                sell_avg = 0
            if symbol == "BANKNIFTY23O1144500CE2023-10-11":
                sell_quantity = 1800
                sell_date = datetime.datetime.strptime(str("2023-10-11 00:00:00"), "%Y-%m-%d %H:%M:%S").date()
                sell_avg = 16.900
            if symbol =="NIFTY23O1919650CE":
                sell_quantity = 1500
                sell_date = datetime.datetime.strptime(str("2023-10-11 00:00:00"), "%Y-%m-%d %H:%M:%S").date()
                sell_avg = 16.900

            all_dict[index]['sell_qty'] = sell_quantity
            all_dict[index]['sell_date'] = sell_date

            all_dict[index]['sell_avg'] = sell_avg
            index += 1
        return all_dict

    def combine_positional(data):
        group_symbol = data.groupby('symbol')
        index = 0
        all_dict = {}
        for symbol in group_symbol.groups:
            all_dict[index] = {}
            all_dict[index]['SYMBOL'] = symbol
            symbol_df = group_symbol.get_group(symbol)
            group_symbol_transact = symbol_df.groupby('trade_type')

            try:
                symbol_buy_df = group_symbol_transact.get_group('buy')
                buy_quantity = symbol_buy_df['quantity'].sum()
                buy_date = datetime.datetime.strptime(str(symbol_buy_df["trade_date"].to_list()[0]),
                                                      "%Y-%m-%d %H:%M:%S").date()
                buy_sum = (symbol_buy_df['price'] * symbol_buy_df['quantity']).sum()
                buy_avg = np.round(buy_sum / buy_quantity, 3)
                all_dict[index]['buy_qty'] = buy_quantity
                all_dict[index]['buy_date'] = buy_date
                all_dict[index]['buy_avg'] = buy_avg
            except:
                print("error")
            try:
                symbol_sell_df = group_symbol_transact.get_group('sell')
                sell_quantity = symbol_sell_df['quantity'].sum()
                print(symbol)
                print(symbol_sell_df["trade_date"].to_list())
                sell_date = datetime.datetime.strptime(str(symbol_sell_df["trade_date"].to_list()[-1]),
                                                       "%Y-%m-%d %H:%M:%S").date()
                sell_sum = (symbol_sell_df['price'] * symbol_sell_df['quantity']).sum()
                sell_avg = np.round(sell_sum / sell_quantity, 3)
            except:
                sell_quantity = 0
                sell_date = "1995-10-03"
                sell_avg = 0
            all_dict[index]['sell_qty'] = sell_quantity
            all_dict[index]['sell_date'] = sell_date

            all_dict[index]['sell_avg'] = sell_avg
            index += 1
        return all_dict

    if fl is not None:
        file_extension = fl.name.split('.')[-1].lower()

        if file_extension == 'csv':
            # Read CSV file directly from the bytes
            df = pd.read_csv(fl, encoding="ISO-8859-1")
            invested_amount = int(st.number_input("Invested Amount"))
            if invested_amount == 0.00:
                invested_amount = 220000

            col1, col2 = st.columns(2)
            df["symboldate"] = df["symbol"] + df["trade_date"]
            df["trade_date"] = pd.to_datetime(df["trade_date"])

            # df = df[(df["trade_date"] >= date1) & (df["trade_date"] <= date2)].copy()

            ALL_DICT_intraday = combine_intraday(df.loc[~df["symbol"].isin(positional_trades)])
            ALL_DICT_posit = combine_positional(df.loc[df["symbol"].isin(positional_trades)])

            intra_df = pd.DataFrame.from_dict(ALL_DICT_intraday, orient='index')
            posit_df = pd.DataFrame.from_dict(ALL_DICT_posit, orient='index')

            df = pd.concat([intra_df, posit_df], ignore_index=True)
            df["sell_date"] = pd.to_datetime(df["sell_date"])
            df.to_csv("test.csv")

            df["sell_date"] = pd.to_datetime(df["sell_date"]).dt.date
            df["buy_date"] = pd.to_datetime(df["buy_date"]).dt.date

            # Initialize Firebase with your credentials
            import json
            key_dict = json.loads(st.secrets["textkey"])
            cred = credentials.Certificate(key_dict)

            try:
                app = firebase_admin.initialize_app(cred)
            except ValueError as e:
                app = firebase_admin.get_app()
            db = firestore.client()
            # Iterate through the DataFrame and add each row as a document
            for index, row in df.iterrows():
                doc_ref = db.collection("trading_logs").document(row["SYMBOL"] + str(row["buy_date"]).replace("/", "-"))
                print(row["SYMBOL"] + str(row["buy_date"]))
                print(row["sell_date"])
                doc_ref.set({"SYMBOL": row["SYMBOL"], "buy_qty": row["buy_qty"],

                             "buy_date": datetime.datetime.combine(row["buy_date"], datetime.time.min),
                             "buy_avg": row["buy_avg"],
                             "sell_qty": row["sell_qty"],

                             "sell_date": datetime.datetime.combine(row["sell_date"], datetime.time.min),
                             "sell_avg": row["sell_avg"], "P/L": 0,
                             "Point_Difference": 0, "Percentage_Difference": 0,
                             "amount_used": 0})

            # Optionally, you can delete the app after using it if it's not needed elsewhere.
            firebase_admin.delete_app(app)


def page2():
    if not util.check_password():
        st.stop()
    trades = util.get_data_firebase()
    # st.set_page_config(page_title="Financial Report", page_icon=":moneybag:", layout="wide")
    st.markdown(styles.header_style, unsafe_allow_html=True)
    st.title("Financial Report")

    df = util.get_data_firebase()
    invested_amount = st.number_input("Invested Amount", 0.00, 1000000.00, 320000.00)
    col1, col2 = st.columns(2)
    df["buy_date"] = pd.to_datetime(df["buy_date"], utc=True)
    df["sell_date"] = pd.to_datetime(df["sell_date"], utc=True)
    startDate = pd.to_datetime(df["buy_date"]).min()
    endDate = pd.to_datetime(df["buy_date"]).max()
    with col1:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))
    with col2:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))
    df_1 = df[df['buy_qty'] != df['sell_qty']].copy()
    df = df[df['buy_qty'] == df['sell_qty']].copy()
    date1 = pd.Timestamp(date1, tz='UTC')
    date2 = pd.Timestamp(date2, tz='UTC')
    df = df[(df["sell_date"] >= date1) & (df["sell_date"] <= date2)].copy()
    # df[df['BUY_QUANTITY'] == df['SELL_QUANTITY']] # Closed Positions
    df['buy_qty'] = df['buy_qty'].astype(float)
    df['buy_avg'] = df['buy_avg'].astype(float)
    df['sell_qty'] = df['sell_qty'].astype(float)
    df['sell_avg'] = df['sell_avg'].astype(float)
    df['P/L'] = (df['sell_qty'] * df['sell_avg']) - (df['buy_qty'] * df['buy_avg'])
    df['Point_Difference'] = (df['sell_avg']) - (df['buy_avg'])
    df['Percentage_Difference'] = ((df['sell_qty'] * df['sell_avg']) - (df['buy_qty'] * df['buy_avg'])) / (
            df['buy_qty'] * df['buy_avg']) * 100
    df["amount_used"] = df['buy_qty'] * df['buy_avg']

    if df_1.shape[0] > 0:
        amount_on_open_trade = (df_1['buy_qty'] * df_1['buy_avg']).sum()
    else:
        amount_on_open_trade = 0
    total_profit = df['P/L'].sum()
    max_loss = round(df['P/L'].min(), 2)
    max_loss = max_loss if max_loss < 0 else 0
    max_profit = round(df['P/L'].max(), 2)
    win_count = (df['P/L'] > 0).sum()
    loss_count = (df['P/L'] <= 0).sum()
    daily_profit = df.groupby('sell_date')['P/L'].sum().reset_index()
    cumulative_profit = daily_profit['P/L'].cumsum()  # Calculate cumulative profit
    max_daily_amount_used = df.groupby('sell_date')['amount_used'].max().reset_index()
    color_map = {True: 'green', False: 'red'}
    figure = {
        'data': [
            go.Scatter(
                x=daily_profit['sell_date'],
                y=cumulative_profit,  # Use cumulative profit data
                mode='lines',
                name='Cumulative Profit',
                line=dict(color='blue'),  # Color of the line plot
            ),
            go.Scatter(
                x=daily_profit['sell_date'],
                y=cumulative_profit,  # Use cumulative profit data
                mode='markers',
                name='Cumulative Profit',
                marker=dict(
                    size=12,  # Increase the size of scatter plot markers
                    color=[color_map[val >= 0] for val in daily_profit['P/L']]
                ),
                showlegend=False,  # Remove the legend entry for this trace
            ),
            go.Bar(
                x=daily_profit['sell_date'],
                y=daily_profit['P/L'],  # Daily profit data
                name='Daily Profit',
                marker=dict(
                    color=[color_map[val >= 0] for val in daily_profit['P/L']]
                ),
            ),
        ],
        'layout': {
            'xaxis': {'title': 'Date'},
            'showlegend': False,
            'title': 'Daily Profit Trend',
            'title_x': 0.5,
        },
    }

    figure_1 = px.scatter(
        max_daily_amount_used,
        x='sell_date',
        y='amount_used',

        title='Maximum Amount Used on Each Day',
    ).update_traces(marker=dict(size=12, color='red'), mode='lines+markers').update_layout(title_x=0.5)
    figure_1.update_layout(xaxis_type='date')
    current_amount = (invested_amount + total_profit) - amount_on_open_trade
    percentage_needed, required_profit = util.required_profit_to_break_even(invested_amount - amount_on_open_trade,
                                                                            current_amount)
    percentage_needed_posit, required_profit_posit = util.required_profit_to_break_even(
        invested_amount - current_amount,
        amount_on_open_trade)
    percentage_needed = 0 if percentage_needed < 0 else percentage_needed
    required_profit = 0 if required_profit < 0 else required_profit
    max_daily_amount_used = df.groupby('buy_date')['amount_used'].max().reset_index()
    util.empty_lines(1)
    st.markdown(styles.metric_style, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([.75, .75, 1])
    with col1:
        metrics = {
            "Invested Amount": [invested_amount, ""],
            "% Required to BreakEven": [percentage_needed, ""],
            "Amt on Open Trade": [amount_on_open_trade, ""],
            "Max Loss": ["", max_loss],
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
                round(current_amount, 2), util.percentage_difference(current_amount, invested_amount)],
            "Amount Required to BreakEven": [required_profit, ""],
            "Expected Return": [percentage_needed_posit, ""],
            "Max Profit": [max_profit, ""],
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
    with col3:
        labels = ['Wins', 'Losses']
        values = [win_count, loss_count]
        colors = ['green', 'red']

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker=dict(colors=colors))])
        st.plotly_chart(fig)
    util.empty_lines(3)
    st.plotly_chart(figure, use_container_width=True)
    util.empty_lines(3)
    st.plotly_chart(figure_1, use_container_width=True)
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
    col_1, col_2 = st.columns([.95, .85])
    with col_1:
        st.dataframe(
            df[["SYMBOL", "buy_date", "buy_qty", "buy_avg", "sell_date","sell_avg", "P/L", "Percentage_Difference", ]].reset_index(
                drop=True))

    with col_2:
        st.dataframe(df_1[["SYMBOL", "buy_date", "buy_qty", "buy_avg", ]].reset_index(drop=True),
                     use_container_width=False)


page_names_to_funcs = {
    "DashBoard": page2,
    "Data Uploader": main_page,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
