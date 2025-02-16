import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import ast 
import re
from bs4 import BeautifulSoup
import pandas as pd

# Title of the web app
st.title("RAG System - Report Analysis and Generating Insights")

# Load external CSS
css_file = 'styles.css'
with open(css_file, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Display an image
image_path = 'Data-Insight.jpg'  # Ensure the image exists in the working directory
st.image(image_path, use_container_width=True, caption="Background Image")

# Query Input Section
st.markdown("## 📌 Ask Your Query")

# User query input field
user_query = st.text_input("Enter your query about the report:", key="query_input")

def display_html_table(html_table: str):
    """
    This function takes an HTML table as a string, parses it, and displays it as a formatted table in Streamlit.
    
    Args:
        html_table (str): The HTML table as a string to be parsed and displayed.
    """
    # Parse the HTML table using BeautifulSoup
    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table')  # Get the first table
    # Extract headers and rows
    rows = table.find_all('tr')
    data = []
    columns = []
    for idx, row in enumerate(rows):
        cells = row.find_all(['th', 'td'])
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        if idx == 0:
            # First row is the header
            columns = cell_texts
        else:
            data.append(cell_texts)

    # Convert to DataFrame
    if columns:
        df = pd.DataFrame(data, columns=columns)
        st.write("**Table Content:**")
        st.dataframe(df)  # Display the table in Streamlit
    else:
        st.write("**No valid table found.**")

# Function to display base64 image
def display_base64_image(base64_code):
    """Converts a base64 string to an image using PIL and displays it in Streamlit."""
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    # Convert the binary data to an image using PIL
    image = Image.open(BytesIO(image_data))
    # Display the image in Streamlit
    st.image(image)

# Button to send the query
if st.button("Submit Query"):
    if user_query.strip():  # Ensure it's not empty
        try:
            response = requests.post("http://127.0.0.1:8000/process_query", json={"query": user_query})
            if response.status_code == 200:
                result = response.json()
                chunk_numbers = []
                image_numbers = []
                # Access the 'insight' key
                if 'insight' in result:
                    insight = result['insight']
                    # Display the result text (answer) from inside 'insight'
                    if 'response' in insight:
                        st.write("### Answer:")
                        st.write(insight['response'])
                        # Regex to extract chunk numbers (handles all three formats)
                        text_matches = re.findall(r"Relevant Chunks[:\s]*\{?([0-9,\s]+)\}?|Relevant Chunks[:\s]*([0-9]+)", insight['response'])
                        if text_matches:
                            for match in text_matches:
                                if match[0]:  # Matches case with curly braces or numbers separated by commas
                                    chunk_numbers.extend(int(num.strip()) for num in match[0].split(",") if num.strip().isdigit())
                                elif match[1]:  # Matches case with just a single number
                                    chunk_numbers.append(int(match[1]))

                            print("Extracted Chunk Numbers:", chunk_numbers)
                            st.write("### Context Texts:")
                        else:
                            chunk_numbers = []  # If no chunk numbers found
                            print("No chunk numbers found.")

                        # Regex to extract image numbers
# Updated regex to allow spaces after commas while keeping the original structure
                        image_matches = re.findall(r"Relevant Images[:\s]*\{?([0-9,\s]+)\}?|Relevant Images[:\s]*([0-9]+)", insight['response'])
# Extract image numbers
                        if image_matches:
                            for match in image_matches:
                                if match[0]:  # Matches case with curly braces or numbers separated by commas
                                    image_numbers.extend(int(num.strip()) for num in match[0].split(",") if num.strip().isdigit())
                                elif match[1]:  # Matches case with just a single number
                                    image_numbers.append(int(match[1]))

                            print("Extracted Image Numbers:", image_numbers)
                    if 'context' in insight and 'texts' in insight['context']:
                        
                        for i, text in enumerate(insight['context']['texts']):
                            # Print the text and page number
                            if i + 1 in chunk_numbers:
                                # Check if the text looks like HTML (table structure)
                                if text.lstrip().startswith('<table>'): # Adjust based on the actual structure
                                    # Use BeautifulSoup to parse HTML and display the table content
                                    # print(text)
                                    display_html_table(text)
                                else:
                                    # If it's a dictionary-like structure, parse it
                                    try:
                                        my_dict = ast.literal_eval(text)  # Try to evaluate as a dictionary
                                        st.write(f"**Text:** {my_dict['text']}")
                                        if 'page_number' in my_dict['metadata']:
                                            st.write(f"### Page number: {my_dict['metadata']['page_number'] + 1}")
                                        if 'filename' in my_dict['metadata']:
                                            st.write(f"### Filename: {my_dict['metadata']['filename']}")
                                    except (SyntaxError, ValueError):
                                        st.write("**Unable to parse text as a dictionary.**")

                                    # Add separator for clarity
                                    st.write("\n" + "-" * 50 + "\n")
                    # Display the images if they exist inside 'context' in 'insight'
                    if 'context' in insight and 'images' in insight['context']:
                        
                        for i,image_b64 in enumerate(insight['context']['images']):
                            # Only process if the image_b64 is valid
                            if i+1 in image_numbers: 
                                if image_b64:
                                    st.write("### Context Images:")  
                                    display_base64_image(image_b64)
                    else:
                        st.warning("⚠️ No images found in the response.")
                else:
                    st.warning("⚠️ Insight data not found in the response.")
            else:
                st.error("⚠️ Error: Backend did not return a successful response.")
        except requests.exceptions.ConnectionError:
            st.error("🚨 Error: Unable to connect to the backend. Please ensure FastAPI is running.")
    else:
        st.warning("⚠️ Please enter a query before submitting.")
