import streamlit as st

def show(df_client, df_contracts, df_cinteractions, df_usage):
    st.header("Imported Datasets")
    with st.expander("Client Data"):
        st.dataframe(df_client)
    with st.expander("Contracts Data"):
        st.dataframe(df_contracts)
    with st.expander("Interactions Data"):
        st.dataframe(df_cinteractions)
    with st.expander("Usage Data"):
        st.dataframe(df_usage)