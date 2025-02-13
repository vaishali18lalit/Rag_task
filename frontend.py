import streamlit as st
import requests

# Title of the web app
st.title("RAG System - Report Analysis and Generating Insights")

# Load external CSS
css_file = 'styles.css'
with open(css_file, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Display an image
image_path = 'Data-Insight.jpg'  # Ensure the image exists in the working directory
st.image(image_path, use_column_width=True, caption="Background Image")

# Query Input Section
st.markdown("## 📌 Ask Your Query")

# User query input field
user_query = st.text_input("Enter your query about the report:", key="query_input")

# Button to send the query
if st.button("Submit Query"):
    if user_query.strip():  # Ensure it's not empty
        try:
            response = requests.post("http://127.0.0.1:8000/process_query", json={"query": user_query})
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"result: {result}")
            else:
                st.error("⚠️ Error: Backend did not return a successful response.")
        except requests.exceptions.ConnectionError:
            st.error("🚨 Error: Unable to connect to the backend. Please ensure FastAPI is running.")
    else:
        st.warning("⚠️ Please enter a query before submitting.")
