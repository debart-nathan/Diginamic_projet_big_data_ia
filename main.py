import streamlit as st
from data_loader import load_data
from sections import (
    introduction,
    data_overview,
    data_analysis,
    missing_data_strategy,
    modeling,
    recommendation
)

# Sidebar navigation options
sections = [
    "Introduction",
    "Data Overview",
    "Data analysis",
    "Missing Data Strategy",
    "Modeling & interpretability",
    "Recommendation"
]

# Read query parameter
query_params = st.query_params
section_from_url = query_params.get("section", None)

# Determine initial section
selected_section = section_from_url if section_from_url in sections else sections[0]

# Sidebar selection
section = st.sidebar.radio("Select a section:", sections, index=sections.index(selected_section))

# Update URL and rerun if changed
if section != section_from_url:
    st.query_params.update({"section": section})
    st.rerun()

# Load data (cached for performance)
@st.cache_data
def load_all_data():
    return load_data()

df_client, df_contracts, df_cinteractions, df_usage, df_merged = load_all_data()


# Route to section with dynamic animated header

if section == "Introduction":
    introduction.show()
elif section == "Data Overview":
    data_overview.show(df_client, df_contracts, df_cinteractions, df_usage)
elif section == "Data analysis":
    data_analysis.show(df_merged)
elif section == "Missing Data Strategy":
    missing_data_strategy.show(df_merged)
elif section ==  "Modeling & interpretability":
    modeling.show(df_merged)
elif section == "Recommendation":
    recommendation.show()