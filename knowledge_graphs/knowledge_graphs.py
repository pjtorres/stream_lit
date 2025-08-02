from fuzzywuzzy import process
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
import spacy
import numpy as np
from collections import Counter, defaultdict
import string
import json

# Helper function to clean the user query
def preprocess_query(query):
    return query.translate(str.maketrans('', '', string.punctuation)).strip().lower()

# Load SpaCy model
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# Enhanced Analytics Functions
class KnowledgeGraphAnalytics:
    def __init__(self, G, data, partition=None):
        self.G = G
        self.data = data
        self.partition = partition
        self.node_attributes = self._compute_node_attributes()
        
    def _compute_node_attributes(self):
        """Compute comprehensive node attributes for analysis"""
        attributes = {}
        
        # Basic centrality measures
        degree_cent = nx.degree_centrality(self.G)
        betweenness_cent = nx.betweenness_centrality(self.G)
        closeness_cent = nx.closeness_centrality(self.G)
        eigenvector_cent = nx.eigenvector_centrality(self.G, max_iter=1000)
        pagerank = nx.pagerank(self.G)
        
        # Structural properties
        clustering = nx.clustering(self.G)
        
        for node in self.G.nodes():
            attributes[node] = {
                'degree': self.G.degree(node),
                'degree_centrality': degree_cent[node],
                'betweenness_centrality': betweenness_cent[node],
                'closeness_centrality': closeness_cent[node],
                'eigenvector_centrality': eigenvector_cent[node],
                'pagerank': pagerank[node],
                'clustering_coefficient': clustering[node],
                'community': self.partition[node] if self.partition else 0
            }
        
        return attributes
    
    def identify_hub_nodes(self, top_n=10):
        """Identify hub nodes using multiple centrality measures"""
        hub_scores = {}
        
        for node in self.G.nodes():
            # Composite hub score combining multiple centralities
            attrs = self.node_attributes[node]
            hub_scores[node] = (
                attrs['degree_centrality'] * 0.3 +
                attrs['betweenness_centrality'] * 0.3 +
                attrs['pagerank'] * 0.2 +
                attrs['eigenvector_centrality'] * 0.2
            )
        
        return sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def find_bridge_nodes(self):
        """Find nodes that connect different communities"""
        bridges = []
        if self.partition:
            for node in self.G.nodes():
                neighbors = list(self.G.neighbors(node))
                node_community = self.partition[node]
                
                # Check if node connects to different communities
                connected_communities = set()
                for neighbor in neighbors:
                    connected_communities.add(self.partition[neighbor])
                
                if len(connected_communities) > 1:
                    bridges.append({
                        'node': node,
                        'community': node_community,
                        'connects_to': list(connected_communities - {node_community}),
                        'bridge_strength': len(connected_communities) - 1,
                        'betweenness': self.node_attributes[node]['betweenness_centrality']
                    })
        
        return sorted(bridges, key=lambda x: x['betweenness'], reverse=True)
    
    def analyze_relationship_patterns(self):
        """Analyze patterns in relationship types"""
        relation_stats = Counter()
        relation_network = defaultdict(list)
        
        for _, row in self.data.iterrows():
            relation = row['relation']
            relation_stats[relation] += 1
            relation_network[relation].append((row['head'], row['tail']))
        
        # Find most connected relationship types
        relation_centrality = {}
        for relation, edges in relation_network.items():
            # Create subgraph for this relation type
            relation_nodes = set()
            for head, tail in edges:
                relation_nodes.add(head)
                relation_nodes.add(tail)
            
            relation_centrality[relation] = {
                'frequency': relation_stats[relation],
                'unique_nodes': len(relation_nodes),
                'density': relation_stats[relation] / len(relation_nodes) if relation_nodes else 0
            }
        
        return relation_centrality
    
    def find_potential_targets(self, seed_nodes, relation_types=None, min_connections=2):
        """Find potential targets based on network proximity to seed nodes"""
        if isinstance(seed_nodes, str):
            seed_nodes = [seed_nodes]
        
        # Find nodes within 2-3 hops of seed nodes
        potential_targets = set()
        
        for seed in seed_nodes:
            if seed in self.G.nodes():
                # Get nodes within 2-3 hops
                for length in [2, 3]:
                    for target in self.G.nodes():
                        if target != seed:
                            try:
                                path_length = nx.shortest_path_length(self.G, seed, target)
                                if path_length == length:
                                    potential_targets.add(target)
                            except nx.NetworkXNoPath:
                                continue
        
        # Score potential targets
        target_scores = {}
        for target in potential_targets:
            attrs = self.node_attributes[target]
            
            # Get relationship types connecting to seed nodes
            connecting_relations = []
            for seed in seed_nodes:
                if seed in self.G.nodes():
                    try:
                        path = nx.shortest_path(self.G, seed, target)
                        for i in range(len(path) - 1):
                            edge_data = self.G.edges[path[i], path[i+1]]
                            connecting_relations.append(edge_data.get('label', 'unknown'))
                    except nx.NetworkXNoPath:
                        continue
            
            # Filter by relation types if specified
            if relation_types:
                if not any(rel in connecting_relations for rel in relation_types):
                    continue
            
            # Filter by minimum connections
            if attrs['degree'] < min_connections:
                continue
                
            target_scores[target] = {
                'pagerank': attrs['pagerank'],
                'degree': attrs['degree'],
                'betweenness': attrs['betweenness_centrality'],
                'connecting_relations': list(set(connecting_relations)),
                'community': attrs['community']
            }
        
        return sorted(target_scores.items(), 
                     key=lambda x: x[1]['pagerank'] * x[1]['degree'], 
                     reverse=True)
    
    def community_analysis(self):
        """Comprehensive community analysis"""
        if not self.partition:
            return None
        
        community_stats = defaultdict(lambda: {
            'nodes': [],
            'size': 0,
            'internal_edges': 0,
            'external_edges': 0,
            'avg_centrality': 0,
            'key_relations': Counter()
        })
        
        # Gather community statistics
        for node in self.G.nodes():
            comm = self.partition[node]
            community_stats[comm]['nodes'].append(node)
            community_stats[comm]['size'] += 1
        
        # Calculate community properties
        for comm_id, stats in community_stats.items():
            nodes = stats['nodes']
            
            # Internal vs external edges
            internal_edges = 0
            external_edges = 0
            
            for node in nodes:
                for neighbor in self.G.neighbors(node):
                    if self.partition[neighbor] == comm_id:
                        internal_edges += 1
                    else:
                        external_edges += 1
            
            stats['internal_edges'] = internal_edges // 2  # Each edge counted twice
            stats['external_edges'] = external_edges
            
            # Average centrality
            centralities = [self.node_attributes[node]['pagerank'] for node in nodes]
            stats['avg_centrality'] = np.mean(centralities)
            
            # Key relations within community
            for node in nodes:
                for neighbor in self.G.neighbors(node):
                    if self.partition[neighbor] == comm_id:
                        edge_data = self.G.edges[node, neighbor]
                        relation = edge_data.get('label', 'unknown')
                        stats['key_relations'][relation] += 1
        
        return dict(community_stats)

# Function to generate the graph (FIXED community filtering)
def generate_graph(data, color_by_community, size_by_centrality, focus_community=None, graph_size="large"):
    # Build the full graph first
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['head'], row['tail'], label=row['relation'])

    # Apply community detection on FULL graph
    partition = None
    community_colors = None
    
    if color_by_community:
        partition = community_louvain.best_partition(G)
        num_communities = len(set(partition.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, num_communities))
        community_colors = {community: rgb2hex(color[:3]) for community, color in enumerate(colors)}

    # NOW apply community filtering if requested
    if focus_community is not None and partition:
        # Debug info
        st.write(f"🔍 Filtering for community {focus_community}")
        
        # Get nodes in the focused community
        focus_nodes = [node for node, comm in partition.items() if comm == focus_community]
        st.write(f"📍 Found {len(focus_nodes)} nodes in community {focus_community}")
        
        if len(focus_nodes) > 0:
            # Include direct neighbors from other communities
            extended_nodes = set(focus_nodes)
            
            for node in focus_nodes:
                neighbors = list(G.neighbors(node))
                for neighbor in neighbors:
                    extended_nodes.add(neighbor)
            
            st.write(f"🔗 Including neighbors, total nodes: {len(extended_nodes)}")
            
            # Create the filtered subgraph
            original_size = len(G.nodes())
            G = G.subgraph(extended_nodes).copy()
            new_size = len(G.nodes())
            
            st.write(f"✂️ Filtered graph: {original_size} → {new_size} nodes")
            
            # Update partition for filtered graph only
            partition = {node: partition[node] for node in G.nodes() if node in partition}
        else:
            st.error(f"❌ No nodes found in community {focus_community}")

    # Set visualization parameters
    if graph_size == "extra_large":
        height, width = "900px", "100%"
        physics_distance = 200
        node_base_size = 25
        spring_length = 300
    elif graph_size == "large":
        height, width = "750px", "100%"
        physics_distance = 160
        node_base_size = 20
        spring_length = 250
    else:  # medium
        height, width = "600px", "100%"
        physics_distance = 130
        node_base_size = 15
        spring_length = 200

    # Create visualization
    net = Network(height=height, width=width, bgcolor="#222222", font_color="white")
    
    # Enhanced physics
    net.set_options(f"""
    var options = {{
      "physics": {{
        "enabled": true,
        "repulsion": {{
          "nodeDistance": {physics_distance}, 
          "centralGravity": 0.1,
          "springLength": {spring_length},
          "springConstant": 0.01,
          "damping": 0.09
        }},
        "solver": "repulsion",
        "stabilization": {{"iterations": 200, "updateInterval": 25}}
      }},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 200,
        "zoomView": true,
        "dragView": true
      }},
      "nodes": {{
        "font": {{"size": 14, "color": "white"}},
        "borderWidth": 2,
        "chosen": {{
          "node": {{
            "label": {{"size": 18, "color": "yellow"}},
            "color": {{"border": "yellow", "background": "rgba(255,255,0,0.3)"}}
          }}
        }}
      }},
      "edges": {{
        "font": {{"size": 10, "color": "white", "strokeWidth": 0, "strokeColor": "black"}},
        "smooth": {{"type": "continuous", "roundness": 0.5}},
        "chosen": {{"edge": {{"color": "yellow", "width": 3}}}}
      }}
    }}
    """)

    # Calculate centrality
    centrality_map = {
        "Degree Centrality": nx.degree_centrality(G),
        "Betweenness Centrality": nx.betweenness_centrality(G),
        "PageRank": nx.pagerank(G)
    }
    centrality = centrality_map.get(size_by_centrality)

    # Add nodes to the graph
    for node in G.nodes():
        if color_by_community and partition and node in partition:
            node_color = community_colors[partition[node]]
            # Highlight focused community nodes
            if focus_community is not None and partition[node] == focus_community:
                border_color = "#ffffff"
                border_width = 4
            else:
                border_color = "#666666"
                border_width = 1
        else:
            node_color = "#97c2fc"
            border_color = "#666666"
            border_width = 1
        
        # Calculate node size
        if centrality:
            node_size = (centrality[node] * 80 + node_base_size)
        else:
            node_size = node_base_size
            
        # Make focused community nodes larger
        if focus_community is not None and partition and node in partition and partition[node] == focus_community:
            node_size *= 1.3
        
        # Create tooltip
        title = f"<b>{node}</b>"
        if color_by_community and partition and node in partition:
            title += f"<br>Community: {partition[node]}"
        title += f"<br>Degree: {G.degree(node)}"
        if centrality:
            title += f"<br>{size_by_centrality}: {centrality[node]:.3f}"
            
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            title += f"<br>Neighbors ({len(neighbors)}): {', '.join(neighbors[:5])}"
            if len(neighbors) > 5:
                title += "..."
        
        net.add_node(
            node, 
            label=str(node), 
            title=title, 
            color=node_color,
            size=node_size,
            borderWidth=border_width,
            borderWidthSelected=6,
            chosen=True
        )

    # Add edges
    for edge in G.edges(data=True):
        label = edge[2].get('label', '')
        
        # Style edges for focused community
        if focus_community is not None and partition:
            node1_comm = partition.get(edge[0])
            node2_comm = partition.get(edge[1])
            
            if node1_comm == focus_community and node2_comm == focus_community:
                edge_color = "#ffffff"
                edge_width = 4
            elif node1_comm == focus_community or node2_comm == focus_community:
                edge_color = "#ff8800"
                edge_width = 3
            else:
                edge_color = "#666666"
                edge_width = 1
        else:
            edge_color = "#848484"
            edge_width = 1
        
        net.add_edge(
            edge[0], 
            edge[1], 
            title=f"Relationship: {label}", 
            label=label,
            color=edge_color,
            width=edge_width,
            font={'size': 10, 'color': 'white'}
        )

    return G, net, partition

# Streamlit App
st.set_page_config(page_title="Knowledge Graph Analytics", layout="wide")
st.title("🧬 Advanced Knowledge Graph Analytics Platform")

st.markdown("""
**Discover actionable insights from your knowledge graph through advanced network analytics, community detection, and intelligent target identification.**
""")

# Initialize session state for focus community
if 'focus_community' not in st.session_state:
    st.session_state.focus_community = None

# Sidebar for file upload and main options
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload Excel file with relationships", type=["xlsx", "xls"])
    
    if uploaded_file:
        st.header("🎨 Visualization Options")
        color_by_community = st.checkbox("Color by Communities", value=True)
        size_by_centrality = st.selectbox(
            "Size nodes by:",
            ["None", "Degree Centrality", "Betweenness Centrality", "PageRank"],
            index=3
        )
        
        graph_size = st.selectbox(
            "Visualization size:",
            ["medium", "large", "extra_large"],
            index=1
        )
        
        st.header("🔍 Analysis Focus")
        analysis_mode = st.selectbox(
            "Choose analysis type:",
            ["Overview Dashboard", "Target Discovery", "Community Analysis", "Relationship Patterns", "Interactive Chat"]
        )

# Main application logic
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    required_columns = ['head', 'tail', 'relation']
    
    if all(col in data.columns for col in required_columns):
        # Generate initial graph to get community info
        G_temp, _, partition_temp = generate_graph(data, color_by_community, size_by_centrality)
        analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
        
        # COMPLETELY FIXED Community selection with button-based approach
        if color_by_community and partition_temp:
            st.sidebar.header("🎯 Community Focus")
            community_stats = analytics.community_analysis()
            
            # Create stable community mapping using actual community IDs
            community_data = []
            for comm_id, stats in community_stats.items():
                if stats['nodes']:
                    representative = max(stats['nodes'], key=lambda x: G_temp.degree(x))
                    community_data.append({
                        'id': comm_id,
                        'representative': representative,
                        'size': stats['size']
                    })
            
            # Sort by size
            community_data.sort(key=lambda x: x['size'], reverse=True)
            
            # Show current focus status
            if st.session_state.focus_community is not None:
                current_item = None
                for item in community_data:
                    if item['id'] == st.session_state.focus_community:
                        current_item = item
                        break
                
                if current_item:
                    st.sidebar.success(f"✅ Currently focused on:\n**{current_item['representative']}**")
                    stats = community_stats[st.session_state.focus_community]
                    st.sidebar.info(f"Community {st.session_state.focus_community}: {stats['size']} nodes, {stats['internal_edges']} internal edges")
                    
                    # Reset button
                    if st.sidebar.button("🔄 Show Full Network", key="reset_to_full"):
                        st.session_state.focus_community = None
                        st.rerun()
                else:
                    st.sidebar.error(f"❌ Community {st.session_state.focus_community} not found!")
            else:
                st.sidebar.info("📊 Currently showing full network")
            
            st.sidebar.markdown("---")
            st.sidebar.write("**Select Community to Focus:**")
            
            # Create community focus buttons
            cols_per_row = 1
            for i in range(0, len(community_data), cols_per_row):
                chunk = community_data[i:i + cols_per_row]
                cols = st.sidebar.columns(len(chunk))
                
                for j, item in enumerate(chunk):
                    with cols[j]:
                        # Show current selection state
                        is_current = st.session_state.focus_community == item['id']
                        
                        button_text = f"{'🎯' if is_current else '📍'} {item['representative']}\n({item['size']} nodes)"
                        button_type = "primary" if is_current else "secondary"
                        
                        if st.button(
                            button_text, 
                            key=f"focus_comm_{item['id']}", 
                            type=button_type,
                            use_container_width=True,
                            disabled=is_current
                        ):
                            st.session_state.focus_community = item['id']
                            st.rerun()
            
            # Alternative: Simple selectbox as backup (no automatic rerun)
            st.sidebar.markdown("---")
            st.sidebar.write("**Or use dropdown:**")
            
            # Create options
            options = ["All Communities"] + [f"{item['representative']} ({item['size']} nodes)" for item in community_data]
            
            # Find current index
            current_index = 0
            if st.session_state.focus_community is not None:
                for i, item in enumerate(community_data):
                    if item['id'] == st.session_state.focus_community:
                        current_index = i + 1
                        break
            
            selected_idx = st.sidebar.selectbox(
                "Community:",
                range(len(options)),
                format_func=lambda x: options[x],
                index=current_index,
                key="community_dropdown",
                label_visibility="collapsed"
            )
            
            # Manual apply button for selectbox
            if st.sidebar.button("Apply Selection", key="apply_selection"):
                if selected_idx == 0:
                    new_focus = None
                else:
                    new_focus = community_data[selected_idx - 1]['id']
                
                if new_focus != st.session_state.focus_community:
                    st.session_state.focus_community = new_focus
                    st.rerun()
        
        # Generate graph with current focus
        G, net, partition = generate_graph(
            data, 
            color_by_community, 
            size_by_centrality, 
            st.session_state.focus_community, 
            graph_size
        )
        
        # Update analytics with filtered graph
        analytics = KnowledgeGraphAnalytics(G, data, partition)
        
        # Analysis modes (Overview Dashboard shown here as example)
        if analysis_mode == "Overview Dashboard":
            st.header("📊 Network Overview Dashboard")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Nodes", len(G.nodes()))
            with col2:
                st.metric("Total Edges", len(G.edges()))
            with col3:
                st.metric("Communities", len(set(partition.values())) if partition else "N/A")
            with col4:
                st.metric("Average Degree", f"{np.mean([G.degree(n) for n in G.nodes()]):.1f}")
            
            # Show current view info
            if st.session_state.focus_community is not None:
                community_stats = analytics.community_analysis()
                stats = community_stats[st.session_state.focus_community]
                representative = max(stats['nodes'], key=lambda x: G.degree(x))
                
                st.subheader(f"🕸️ {representative} Community Focused View")
                st.info(f"Showing {representative} community with direct connections. White borders indicate focused nodes.")
                
                # Reset button
                if st.button("🔄 Return to Full Network", key="reset_focus"):
                    st.session_state.focus_community = None
                    st.rerun()
            else:
                st.subheader("🕸️ Complete Network Visualization")
            
            # Generate and display graph
            import time
            timestamp = int(time.time() * 1000)
            graph_filename = f"kg_graph_{timestamp}.html"
            
            net.save_graph(graph_filename)
            
            try:
                with open(graph_filename, 'r') as f:
                    graph_html = f.read()
                
                # Make HTML unique to force refresh
                unique_id = f"graph_{st.session_state.focus_community}_{timestamp}"
                graph_html = graph_html.replace('<div id="mynetworkid"', f'<div id="{unique_id}"')
                
                height_map = {"medium": 650, "large": 800, "extra_large": 950}
                components.html(graph_html, height=height_map.get(graph_size, 800))
                
            except FileNotFoundError:
                st.error(f"Could not load graph file: {graph_filename}")
            
            # Hub analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Top Hub Nodes")
                hubs = analytics.identify_hub_nodes(10)
                hub_df = pd.DataFrame(hubs, columns=['Node', 'Hub Score'])
                st.dataframe(hub_df, use_container_width=True)
            
            with col2:
                st.subheader("🌉 Bridge Nodes")
                bridges = analytics.find_bridge_nodes()[:10]
                if bridges:
                    bridge_df = pd.DataFrame([{
                        'Node': b['node'],
                        'Communities Connected': b['bridge_strength'],
                        'Betweenness': f"{b['betweenness']:.3f}"
                    } for b in bridges])
                    st.dataframe(bridge_df, use_container_width=True)
                else:
                    st.info("No bridge nodes found")
        
        # Add other analysis modes here...
        
    else:
        st.error(f"Please ensure your file contains columns: {', '.join(required_columns)}")
else:
    st.info("👆 Upload an Excel file to start analyzing your knowledge graph!")
    
    # Sample data format
    st.subheader("📋 Expected Data Format")
    sample_df = pd.DataFrame({
        'head': ['Bifidobacterium', 'B. infantis', 'Probiotic'],
        'tail': ['gut health', 'immune response', 'microbiome'],
        'relation': ['affects', 'modulates', 'influences']
    })
    st.dataframe(sample_df, use_container_width=True)
