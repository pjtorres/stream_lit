from fuzzywuzzy import process
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components  # CORRECT import for 1.48.0
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
import spacy
import numpy as np
import string

# Set page config to wide layout
st.set_page_config(
    page_title="Interactive Knowledge Graphs", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# STREAMLIT 1.48.0 SPECIFIC WIDTH FIX
st.markdown("""
<style>
    /* Remove Streamlit's default padding/margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
        max-width: none !important;
        width: 100% !important;
    }
    
    /* Remove top app margin */
    .stApp > div:first-child {
        margin-top: -80px;
    }
    
    /* Force HTML components to full width */
    .stHtml {
        width: 100% !important;
        max-width: none !important;
    }
    
    /* Force iframe full width */
    iframe {
        width: 100% !important;
        min-width: 100% !important;
        max-width: none !important;
        border: none !important;
    }
    
    /* Fix for Streamlit 1.48+ container width */
    section.main > div {
        width: 100% !important;
        max-width: none !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    
    /* HTML component container fix */
    div[data-testid="stHtml"] {
        width: 100% !important;
        max-width: none !important;
    }
    
    /* Element container fix */
    .element-container {
        width: 100% !important;
        max-width: none !important;
    }
    
    /* Additional fix for wide layout */
    .css-1d391kg, .css-18e3th9, .css-1y4p8pa {
        width: 100% !important;
        max-width: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to clean the user query
def preprocess_query(query):
    return query.translate(str.maketrans('', '', string.punctuation)).strip().lower()

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit App Title
st.title("Interactive Knowledge Graphs")

st.markdown("""
Upload a dataset and explore its knowledge graph interactively. Use additional options to color nodes by Louvain communities or size them by centrality measures.
""")

# DEBUG SECTION - Test width with Streamlit 1.48.0
st.write("**Width Debug Test (Streamlit 1.48.0):**")
st.write(f"Streamlit version: {st.__version__}")

# Test HTML component width with correct import
test_html = """
<div style="background: red; width: 100%; height: 50px; text-align: center; color: white; font-size: 18px; line-height: 50px;">
This red bar should span FULL width - if it doesn't, there's a width constraint
</div>
"""
components.html(test_html, height=60)

# File uploader for mapping file
uploaded_file = st.file_uploader("Upload an excel file containing relationships. ", type=["xlsx", "xls"])

# Enhanced function to generate the graph with width fixes
def generate_graph(data, color_by_community, size_by_centrality):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['head'], row['tail'], label=row['relation'])

    # Create PyVis network with enhanced width settings
    net = Network(
        height="900px", 
        width="100%", 
        notebook=False,
        bgcolor="#ffffff",
        font_color="black"
    )

    # Apply Louvain Community Coloring
    if color_by_community:
        partition = community_louvain.best_partition(G, resolution=1.3, random_state=42)
        num_communities = len(set(partition.values()))
        
        if num_communities <= 10:
            colors = plt.cm.tab10(range(num_communities))
        elif num_communities <= 20:
            colors = plt.cm.tab20(range(num_communities))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, num_communities))
        
        community_colors = {community: rgb2hex(color[:3]) for community, color in enumerate(colors)}
    else:
        partition = None
        community_colors = None

    # Calculate centrality if selected
    if size_by_centrality == "Degree Centrality":
        centrality = nx.degree_centrality(G)
    elif size_by_centrality == "Betweenness Centrality":
        centrality = nx.betweenness_centrality(G)
    elif size_by_centrality == "PageRank":
        centrality = nx.pagerank(G)
    else:
        centrality = None

    # Add nodes to the graph
    for node in G.nodes():
        node_color = community_colors[partition[node]] if color_by_community else "#97c2fc"
        node_size = centrality[node] * 200 if centrality else 35
        net.add_node(
            node,
            label=str(node),
            title=f"Node: {node}" + (f", Community: {partition[node]}" if color_by_community else ""),
            color=node_color,
            size=node_size,
            font={"size": 24}
        )

    # Add edges to the graph
    for edge in G.edges(data=True):
        label = edge[2].get('label', '')
        net.add_edge(edge[0], edge[1], title=label, label=label)

    # Enhanced physics settings for better spread
    net.repulsion(
        node_distance=180,
        central_gravity=0.08,
        spring_length=300,
        spring_strength=0.003,
        damping=0.95
    )
    
    # Add JavaScript options for better interactivity
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      }
    }
    """)

    return G, net, partition

# Function to fix PyVis HTML for full width (Streamlit 1.48.0 specific)
def fix_pyvis_html_for_streamlit_1_48(html_content):
    """
    Inject CSS into PyVis HTML to force full width in Streamlit 1.48.0
    """
    width_fix_css = """
    <style>
    body { 
        margin: 0 !important; 
        padding: 0 !important; 
        width: 100vw !important;
        overflow-x: hidden;
    }
    #mynetworkid { 
        width: 100% !important; 
        min-width: 100% !important;
        height: 900px !important; 
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    """
    
    # Insert CSS after opening body tag or at the beginning
    if '<body' in html_content:
        body_start = html_content.find('>') + 1
        html_content = html_content[:body_start] + width_fix_css + html_content[body_start:]
    else:
        html_content = width_fix_css + html_content
    
    return html_content

# Enhanced display function for Streamlit 1.48.0
def display_graph_full_width(graph_html, height=900):
    """
    Display graph with full width using multiple techniques for Streamlit 1.48.0
    """
    
    # Method 1: Fix the HTML directly
    fixed_html = fix_pyvis_html_for_streamlit_1_48(graph_html)
    
    # Method 2: Use container for proper width
    with st.container():
        components.html(
            fixed_html, 
            height=height + 50,  # Add some extra height
            scrolling=False
        )

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    required_columns = ['head', 'tail', 'relation']

    if all(col in data.columns for col in required_columns):
        # Sidebar options for customization
        st.sidebar.header("Customization Options")
        color_by_community = st.sidebar.checkbox("Color Nodes by Louvain Communities")
        size_by_centrality = st.sidebar.selectbox(
            "Size Nodes by Centrality Measure",
            ["None", "Degree Centrality", "Betweenness Centrality", "PageRank"]
        )

        # Check if the graph needs to be regenerated
        if "graph" not in st.session_state or st.session_state.get("color_by_community") != color_by_community or st.session_state.get("size_by_centrality") != size_by_centrality:
            # Generate and save the graph
            G, net, partition = generate_graph(data, color_by_community, size_by_centrality)
            net.save_graph("knowledge_graph.html")
            with open("knowledge_graph.html", 'r') as f:
                st.session_state["graph_html"] = f.read()
            st.session_state["graph"] = G
            st.session_state["partition"] = partition
            st.session_state["color_by_community"] = color_by_community
            st.session_state["size_by_centrality"] = size_by_centrality
        else:
            G = st.session_state["graph"]
            partition = st.session_state["partition"]

        # SUMMARY SECTION
        st.subheader("Graph Summary")
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        if partition:
            num_communities = len(set(partition.values()))
        else:
            partition = community_louvain.best_partition(G, resolution=1.3, random_state=42)
            num_communities = len(set(partition.values()))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Nodes", num_nodes)
        with col2:
            st.metric("Number of Edges", num_edges)
        with col3:
            st.metric("Number of Communities", num_communities)

        # Display the graph with FULL WIDTH for Streamlit 1.48.0
        st.subheader("Knowledge Graph Visualization")
        
        # Use the enhanced display function
        display_graph_full_width(st.session_state["graph_html"], height=900)
        
        # Alternative method if above doesn't work:
        st.write("**Alternative Display Method:**")
        st.write("If the graph above still appears narrow, try refreshing the page or using a different browser.")
        
        # Simple fallback display
        with st.expander("Fallback Display (click if main graph is still narrow)"):
            components.html(st.session_state["graph_html"], height=800)

    else:
        st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")

# Additional troubleshooting info for Streamlit 1.48.0
st.markdown("""
---
**Troubleshooting for Streamlit 1.48.0:**
- If graph still appears narrow, try refreshing the page
- Check that your browser zoom is at 100%
- Try a different browser (Chrome usually works best)
- Clear browser cache if issues persist
""")
