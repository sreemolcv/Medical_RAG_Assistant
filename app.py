import streamlit as st
from ingest import rag_chain
query = st.text_input(
    "Ask question about stroke"
)

if query:
    answer = rag_chain(query)
    st.write(answer)