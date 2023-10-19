import hmac
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import pandas as pd


def check_password():
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


def empty_lines(n: int) -> None:
    for _ in range(n):
        st.write("")


def percentage_difference(value_1, value_2):
    return round(((value_1 - value_2) / ((value_1 + value_2) / 2)) * 100, 2)


def required_profit_to_break_even(original_amount, amount):
    profit = original_amount - amount
    percentage_profit_needed = (profit / amount) * 100
    return round(percentage_profit_needed, 2), round(profit)


def initialize_firebase():
    import json
    key_dict = json.loads(st.secrets["textkey"])
    cred = credentials.Certificate(key_dict)
    try:
        return firebase_admin.initialize_app(cred)
    except ValueError as e:
        return firebase_admin.get_app()


def get_data_firebase():
    app = initialize_firebase()
    db = firestore.client()
    data_dict = {}
    users_ref = db.collection("trading_logs")
    docs = users_ref.stream()
    for doc in docs:
        data_dict[doc.id] = doc.to_dict()
    return pd.DataFrame(data_dict).T


def custom_function(row):
    # Guard against division by zero
    denominator = row['A'] - row['B']
    if denominator == 0:
        result = "Undefined"  # or any other value that you see fit
    else:
        result = 1.0 + round((row['C'] - row['A']) / denominator, 2)
        result = f"{result:.2f}"
    return f"1:{result}"
