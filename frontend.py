import streamlit as st
import requests
from datetime import datetime
import pandas as pd

# Configuration
API_URL = "http://localhost:8000"

# Helper functions
def search_emails(query: str, account: str = None, folder: str = None, category: str = None):
    params = {
        "text": query,
        "account": account,
        "folder": folder,
        "category": category
    }
    response = requests.get(f"{API_URL}/emails/search", params=params)
    return response.json()

def suggest_reply(email: dict):
    response = requests.post(f"{API_URL}/emails/suggest-reply", json=email)
    return response.json().get("reply", "")

# Streamlit UI
st.set_page_config(layout="wide")
st.title("ReachInbox Email Aggregator")

# Sidebar with filters
with st.sidebar:
    st.header("Filters")
    search_query = st.text_input("Search emails")
    account_filter = st.selectbox("Account", ["All", "account1@example.com", "account2@example.com"])
    folder_filter = st.selectbox("Folder", ["INBOX", "SENT", "SPAM"])
    category_filter = st.selectbox("Category", ["All", "Interested", "Meeting Booked", "Not Interested", "Spam", "Out of Office"])

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Emails")
    
    # Apply filters
    filters = {
        "text": search_query if search_query else None,
        "account": account_filter if account_filter != "All" else None,
        "folder": folder_filter if folder_filter != "All" else None,
        "category": category_filter if category_filter != "All" else None
    }
    
    emails = search_emails(**filters)
    
    if emails:
        email_df = pd.DataFrame([{
            "From": e["from_"],
            "Subject": e["subject"],
            "Date": datetime.strptime(e["date"], "%Y-%m-%dT%H:%M:%S").strftime("%b %d, %H:%M"),
            "Category": e.get("category", "Uncategorized")
        } for e in emails])
        
        st.dataframe(
            email_df,
            column_config={
                "From": st.column_config.TextColumn("From", width="medium"),
                "Subject": st.column_config.TextColumn("Subject", width="large"),
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Category": st.column_config.TextColumn("Category", width="small")
            },
            hide_index=True,
            use_container_width=True
        )
        
        selected_email = st.selectbox(
            "View email",
            [f"{e['subject']} - {e['from_']}" for e in emails]
        )
        
        if selected_email:
            selected_index = next(i for i, e in enumerate(emails) 
                                if f"{e['subject']} - {e['from_']}" == selected_email)
            selected_email = emails[selected_index]
    else:
        st.info("No emails found matching your criteria")

with col2:
    if 'selected_email' in locals():
        st.subheader("Email Details")
        st.markdown(f"**From:** {selected_email['from_']}")
        st.markdown(f"**To:** {selected_email['to']}")
        st.markdown(f"**Date:** {selected_email['date']}")
        st.markdown(f"**Category:** {selected_email.get('category', 'Uncategorized')}")
        st.markdown(f"**Subject:** {selected_email['subject']}")
        
        st.subheader("Content")
        st.text_area("Body", value=selected_email['body'], height=200)
        
        if st.button("Generate Reply"):
            reply = suggest_reply(selected_email)
            st.subheader("Suggested Reply")
            st.text_area("Reply", value=reply, height=200)
            
            if st.button("Copy to Clipboard"):
                st.session_state.clipboard = reply
                st.success("Reply copied to clipboard!")
