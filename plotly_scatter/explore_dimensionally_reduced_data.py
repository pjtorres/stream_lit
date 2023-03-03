"""
Created on 03-03-2023
@author: Pedro J. Torres
"""

import streamlit as st
import pandas as pd
import base64
import pandas as pd
import plotly.express as px


st.title("Explore Dimensionally  Reduced Data")

st.markdown("""
Here you can make a variaty of scatter plots based on the loadings from dementional reduciton approaches. Currently you have loadings from two approaches:
* gUniFrac (species)
* PCA (functional annotation)

**If you want to look at the gUniFrac taxa clusters select column 'cluster' and select feature type 'taxa'.**

**If you want to look at the PCA functional clusters select column 'functional_clusters' and select feature type 'function'.**

""")




uploaded_file = st.file_uploader("Upload a mapping file containing coordinate loadings", type="txt")
if uploaded_file is not None:
    # Read the CSV file into a Pandas dataframe
    df = pd.read_csv(uploaded_file, sep='\t')

    #Sidebar
    st.sidebar.title("Dataset")
    file_name = uploaded_file.name
    st.sidebar.text_input("You are using",(file_name))
    # option = st.sidebar.selectbox("which Dashboard?", (list(df.colnames)))
    category = st.sidebar.selectbox('Select a column', df.columns)
    feature = st.sidebar.selectbox('Select feature type', ('taxa','function'))


    if feature == 'function':
        colors = {2: 'green', 1: 'yellow',3:'red',4:'purple'}
        if category =='functional_clusters':
            fig = px.scatter(df, x='PC2_Functional', y='PC1_Functional',
                         color=category, color_discrete_sequence=['yellow','green','purple','red'])
        else:
            fig = px.scatter(df, x='PC2_Functional', y='PC1_Functional',
                         color=category)

    if feature == 'taxa':
        fig = px.scatter(df, x='gUniFrac_PCoA_1', y='gUniFrac_PCoA_2',
                             color=category, color_discrete_sequence=['yellow','green','purple','red'])


    st.plotly_chart(fig)

    # Add a download button for the Plotly HTML
    html = fig.to_html(include_plotlyjs='cdn')
    st.download_button(
        label='Download Plotly HTML',
        data=html,
        file_name=feature+ '_'+category+'_plot.html',
        mime='text/html'
    )
