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

# Streamlit App
st.set_page_config(page_title="Knowledge Graph Analytics", layout="wide")
st.title("🧬 Advanced Knowledge Graph Analytics Platform")

st.markdown("""
**Discover actionable insights from your knowledge graph through advanced network analytics, community detection, and intelligent target identification.**
""")

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
        
        # Graph size selector
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

# Function to generate the graph (enhanced with community filtering)
def generate_graph(data, color_by_community, size_by_centrality, focus_community=None, graph_size="large"):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['head'], row['tail'], label=row['relation'])

    # Set visualization size with better spacing
    if graph_size == "extra_large":
        height, width = "900px", "100%"
        physics_distance = 200  # Increased from 150
        node_base_size = 25
        spring_length = 300     # Increased
    elif graph_size == "large":
        height, width = "750px", "100%"
        physics_distance = 160  # Increased from 120
        node_base_size = 20
        spring_length = 250     # Increased
    else:  # medium
        height, width = "600px", "100%"
        physics_distance = 130  # Increased from 100
        node_base_size = 15
        spring_length = 200     # Increased

    net = Network(height=height, width=width, bgcolor="#222222", font_color="white")
    
    # Enhanced physics for better layout and more spread
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
    """);

    # Apply Louvain Community Coloring
    partition = None
    community_colors = None
    
    if color_by_community:
        partition = community_louvain.best_partition(G)
        num_communities = len(set(partition.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, num_communities))
        community_colors = {community: rgb2hex(color[:3]) for community, color in enumerate(colors)}

    # Filter graph for specific community if requested
    if focus_community is not None and partition:
        # Get nodes in the focused community
        focus_nodes = [node for node, comm in partition.items() if comm == focus_community]
        
        # Also include directly connected nodes from other communities (1-hop neighbors)
        extended_nodes = set(focus_nodes)
        for node in focus_nodes:
            for neighbor in G.neighbors(node):
                extended_nodes.add(neighbor)
        
        # Create subgraph
        G_filtered = G.subgraph(extended_nodes).copy()
        
        # Update partition for filtered graph
        partition_filtered = {node: partition[node] for node in G_filtered.nodes()}
        
        G = G_filtered
        partition = partition_filtered

    # Calculate centrality
    centrality_map = {
        "Degree Centrality": nx.degree_centrality(G),
        "Betweenness Centrality": nx.betweenness_centrality(G),
        "PageRank": nx.pagerank(G)
    }
    centrality = centrality_map.get(size_by_centrality)

    # Add nodes to the graph
    for node in G.nodes():
        if color_by_community and partition:
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
            node_size = (centrality[node] * 80 + node_base_size)  # Reduced multiplier from 150 to 80
        else:
            node_size = node_base_size
            
        # Make focused community nodes larger
        if focus_community is not None and partition and partition[node] == focus_community:
            node_size *= 1.2  # Reduced from 1.3
        
        # Create detailed tooltip
        title = f"<b>{node}</b>"
        if color_by_community and partition:
            title += f"<br>Community: {partition[node]}"
        title += f"<br>Degree: {G.degree(node)}"
        if centrality:
            title += f"<br>{size_by_centrality}: {centrality[node]:.3f}"
            
        # Add neighbors info
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

    # Add edges with enhanced styling
    for edge in G.edges(data=True):
        label = edge[2].get('label', '')
        
        # Style edges differently for focused community
        if focus_community is not None and partition:
            node1_comm = partition[edge[0]]
            node2_comm = partition[edge[1]]
            
            if node1_comm == focus_community and node2_comm == focus_community:
                # Internal edges in focused community
                edge_color = "#ffffff"
                edge_width = 3
            elif node1_comm == focus_community or node2_comm == focus_community:
                # Edges connecting to focused community
                edge_color = "#ffaa00"
                edge_width = 2
            else:
                # External edges
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

# Main application logic
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    required_columns = ['head', 'tail', 'relation']
    
    if all(col in data.columns for col in required_columns):
        # Generate initial graph to get community info
        G_temp, _, partition_temp = generate_graph(data, color_by_community, size_by_centrality)
        analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
        
        # Community selection for focused view
        focus_community = None
        if color_by_community and partition_temp:
            st.sidebar.header("🎯 Community Focus")
            community_stats = analytics.community_analysis()
            
            # Create community options with sizes
            community_options = ["All Communities (Full Graph)"]
            for comm_id, stats in sorted(community_stats.items(), key=lambda x: x[1]['size'], reverse=True):
                community_options.append(f"Community {comm_id} ({stats['size']} nodes)")
            
            selected_community = st.sidebar.selectbox(
                "Focus on specific community:",
                community_options,
                help="Select a community to zoom in and see detailed connections"
            )
            
            if selected_community != "All Communities (Full Graph)":
                focus_community = int(selected_community.split()[1])
                st.sidebar.success(f"Focusing on Community {focus_community}")
                
                # Show community info
                if focus_community in community_stats:
                    stats = community_stats[focus_community]
                    st.sidebar.write(f"**Community {focus_community} Details:**")
                    st.sidebar.write(f"- Nodes: {stats['size']}")
                    st.sidebar.write(f"- Internal edges: {stats['internal_edges']}")
                    st.sidebar.write(f"- External edges: {stats['external_edges']}")
        
        # Generate final graph with focus
        G, net, partition = generate_graph(data, color_by_community, size_by_centrality, focus_community, graph_size)
        
        # Analysis modes
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
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Top Hub Nodes")
                hubs = analytics.identify_hub_nodes(10)
                hub_df = pd.DataFrame(hubs, columns=['Node', 'Hub Score'])
                st.dataframe(hub_df, use_container_width=True)
                
                # Simple matplotlib chart
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(hub_df['Node'], hub_df['Hub Score'])
                ax.set_xlabel('Hub Score')
                ax.set_title('Top Hub Nodes')
                plt.tight_layout()
                st.pyplot(fig)
            
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
                    st.info("No bridge nodes found (requires community detection)")
            
            # Network visualization with focus indicator
            if focus_community is not None:
                st.subheader(f"🕸️ Community {focus_community} Focused View")
                st.info(f"Showing Community {focus_community} with its direct connections. White borders indicate focused community nodes.")
            else:
                st.subheader("🕸️ Complete Network Visualization")
            
            # Add reset button when focused
            if focus_community is not None:
                if st.button("🔄 Return to Full Network View"):
                    st.experimental_rerun()
            
            net.save_graph("temp_graph.html")
            with open("temp_graph.html", 'r') as f:
                graph_html = f.read()
            
            # Adjust height based on graph size
            height_map = {"medium": 650, "large": 800, "extra_large": 950}
            components.html(graph_html, height=height_map.get(graph_size, 800))
            
            # Relationship analysis
            st.subheader("🔗 Relationship Type Analysis")
            relation_stats = analytics.analyze_relationship_patterns()
            relation_df = pd.DataFrame([
                {
                    'Relation': rel,
                    'Frequency': stats['frequency'],
                    'Unique Nodes': stats['unique_nodes'],
                    'Density': f"{stats['density']:.3f}"
                }
                for rel, stats in relation_stats.items()
            ]).sort_values('Frequency', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(relation_df, use_container_width=True)
            with col2:
                # Simple pie chart with matplotlib
                top_relations = relation_df.head(8)
                if len(top_relations) > 0:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.pie(top_relations['Frequency'], labels=top_relations['Relation'], autopct='%1.1f%%')
                    ax.set_title('Top Relationship Types')
                    st.pyplot(fig)
                else:
                    st.info("No relationship data to display")
            with col1:
                st.dataframe(relation_df, use_container_width=True)
            with col2:
                fig = px.pie(relation_df.head(10), values='Frequency', names='Relation',
                            title="Top Relationship Types")
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_mode == "Target Discovery":
            st.header("🎯 Target Discovery Engine")
            
            st.markdown("""
            **Find potential therapeutic targets based on network proximity and centrality measures.**
            Enter seed nodes (e.g., 'Bifidobacterium', 'B. infantis') to discover related targets.
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                seed_input = st.text_input(
                    "Enter seed nodes (comma-separated):",
                    placeholder="Bifidobacterium, B. infantis, probiotic"
                )
                
            with col2:
                min_connections = st.slider("Minimum connections:", 1, 10, 2)
            
            # Relation type filter
            all_relations = list(set(data['relation'].values))
            selected_relations = st.multiselect(
                "Filter by relationship types (optional):",
                all_relations,
                help="Leave empty to include all relationship types"
            )
            
            if seed_input:
                seed_nodes = [node.strip() for node in seed_input.split(',')]
                
                # Fuzzy match seed nodes to actual graph nodes
                matched_seeds = []
                for seed in seed_nodes:
                    matches = process.extract(seed.lower(), 
                                            [n.lower() for n in G.nodes()], 
                                            limit=3)
                    if matches and matches[0][1] > 60:  # 60% similarity threshold
                        original_node = [n for n in G.nodes() if n.lower() == matches[0][0]][0]
                        matched_seeds.append(original_node)
                        st.success(f"Matched '{seed}' to '{original_node}'")
                    else:
                        st.warning(f"No close match found for '{seed}'")
                
                if matched_seeds:
                    # Find potential targets
                    targets = analytics.find_potential_targets(
                        matched_seeds, 
                        selected_relations if selected_relations else None,
                        min_connections
                    )
                    
                    if targets:
                        st.subheader(f"🎯 Top Potential Targets (Found {len(targets)})")
                        
                        # Create detailed results
                        target_results = []
                        for target, scores in targets[:20]:  # Top 20
                            target_results.append({
                                'Target': target,
                                'PageRank': f"{scores['pagerank']:.4f}",
                                'Degree': scores['degree'],
                                'Betweenness': f"{scores['betweenness']:.4f}",
                                'Community': scores['community'],
                                'Key Relations': ', '.join(scores['connecting_relations'][:3])
                            })
                        
                        target_df = pd.DataFrame(target_results)
                        st.dataframe(target_df, use_container_width=True)
                        
                        # Visualize top targets with matplotlib
                        top_targets = targets[:10]
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        x_vals = [t[1]['pagerank'] for t in top_targets]
                        y_vals = [t[1]['degree'] for t in top_targets]
                        sizes = [t[1]['betweenness'] * 1000 + 50 for t in top_targets]
                        colors = [t[1]['community'] for t in top_targets]
                        
                        scatter = ax.scatter(x_vals, y_vals, s=sizes, c=colors, 
                                           cmap='viridis', alpha=0.7, edgecolors='black')
                        
                        # Add labels for top targets
                        for i, (target, _) in enumerate(top_targets):
                            label = target[:15] + '...' if len(target) > 15 else target
                            ax.annotate(label, (x_vals[i], y_vals[i]), 
                                      xytext=(5, 5), textcoords='offset points', 
                                      fontsize=8, alpha=0.8)
                        
                        ax.set_xlabel('PageRank Score')
                        ax.set_ylabel('Node Degree')
                        ax.set_title('Potential Targets: PageRank vs Degree (Size = Betweenness)')
                        
                        # Add colorbar
                        cbar = plt.colorbar(scatter)
                        cbar.set_label('Community')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("No potential targets found with the specified criteria.")
        
        elif analysis_mode == "Community Analysis":
            st.header("🏘️ Community Structure Analysis")
            
            if partition:
                community_stats = analytics.community_analysis()
                
                # Community overview
                st.subheader("Community Overview")
                comm_overview = []
                for comm_id, stats in community_stats.items():
                    comm_overview.append({
                        'Community': comm_id,
                        'Size': stats['size'],
                        'Internal Edges': stats['internal_edges'],
                        'External Edges': stats['external_edges'],
                        'Modularity': stats['internal_edges'] / (stats['internal_edges'] + stats['external_edges'] + 0.001),
                        'Avg PageRank': f"{stats['avg_centrality']:.4f}"
                    })
                
                comm_df = pd.DataFrame(comm_overview).sort_values('Size', ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(comm_df, use_container_width=True)
                
                with col2:
                    # Simple scatter plot with matplotlib
                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(comm_df['Size'], comm_df['Modularity'], 
                                       c=range(len(comm_df)), cmap='viridis', alpha=0.7)
                    ax.set_xlabel('Community Size')
                    ax.set_ylabel('Modularity')
                    ax.set_title('Community Size vs Modularity')
                    
                    # Add community labels
                    for i, row in comm_df.iterrows():
                        ax.annotate(f"C{row['Community']}", 
                                  (row['Size'], row['Modularity']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                    st.pyplot(fig)
                
                # Quick community focus buttons
                st.subheader("🎯 Quick Community Focus")
                cols = st.columns(min(5, len(community_stats)))
                top_communities = sorted(community_stats.items(), key=lambda x: x[1]['size'], reverse=True)[:5]
                
                for i, (comm_id, stats) in enumerate(top_communities):
                    with cols[i]:
                        if st.button(f"Focus on Community {comm_id}\n({stats['size']} nodes)", key=f"focus_{comm_id}"):
                            # Store focus community in session state and rerun
                            st.session_state.focus_community = comm_id
                            st.experimental_rerun()
                
                # Detailed community analysis
                selected_community = st.selectbox(
                    "Select community for detailed analysis:",
                    options=list(community_stats.keys()),
                    format_func=lambda x: f"Community {x} ({community_stats[x]['size']} nodes)"
                )
                
                if selected_community is not None:
                    st.subheader(f"Community {selected_community} Details")
                    stats = community_stats[selected_community]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nodes", stats['size'])
                    with col2:
                        st.metric("Internal Connections", stats['internal_edges'])
                    with col3:
                        st.metric("External Connections", stats['external_edges'])
                    
                    # Top nodes in community
                    st.write("**Key nodes in this community:**")
                    community_nodes = []
                    for node in stats['nodes']:
                        attrs = analytics.node_attributes[node]
                        community_nodes.append({
                            'Node': node,
                            'Degree': attrs['degree'],
                            'PageRank': f"{attrs['pagerank']:.4f}",
                            'Betweenness': f"{attrs['betweenness_centrality']:.4f}"
                        })
                    
                    community_node_df = pd.DataFrame(community_nodes).sort_values('PageRank', ascending=False)
                    st.dataframe(community_node_df.head(10), use_container_width=True)
                    
                    # Key relationships
                    st.write("**Key relationships within community:**")
                    relations_list = [(rel, count) for rel, count in stats['key_relations'].most_common(10)]
                    if relations_list:
                        rel_df = pd.DataFrame(relations_list, columns=['Relationship', 'Count'])
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(rel_df['Relationship'], rel_df['Count'])
                        ax.set_xlabel('Count')
                        ax.set_title('Key Relationships in Community')
                        plt.tight_layout()
                        st.pyplot(fig)
            else:
                st.info("Community analysis requires community detection to be enabled.")
        
        elif analysis_mode == "Interactive Chat":
            st.header("💬 Knowledge Graph Assistant")
            
            # Enhanced chatbot with more capabilities
            user_question = st.text_input(
                "Ask about your knowledge graph:",
                placeholder="Try: 'Find targets similar to Bifidobacterium', 'What's in the largest community?', 'Show me bridge nodes'"
            )
            
            if user_question:
                question_lower = user_question.lower()
                
                # Enhanced query processing
                if "targets similar to" in question_lower or "similar targets" in question_lower:
                    # Extract entity
                    words = preprocess_query(user_question).split()
                    if "to" in words:
                        entity_idx = words.index("to") + 1
                        if entity_idx < len(words):
                            entity = " ".join(words[entity_idx:])
                            
                            # Find similar targets
                            matches = process.extract(entity, [n.lower() for n in G.nodes()], limit=1)
                            if matches and matches[0][1] > 60:
                                matched_node = [n for n in G.nodes() if n.lower() == matches[0][0]][0]
                                targets = analytics.find_potential_targets([matched_node], min_connections=1)
                                
                                st.write(f"**Targets similar to '{matched_node}':**")
                                for target, scores in targets[:10]:
                                    st.write(f"- **{target}** (PageRank: {scores['pagerank']:.4f}, Degree: {scores['degree']})")
                
                elif "largest community" in question_lower or "biggest community" in question_lower:
                    if partition:
                        community_stats = analytics.community_analysis()
                        largest_comm = max(community_stats.items(), key=lambda x: x[1]['size'])
                        comm_id, stats = largest_comm
                        
                        st.write(f"**Largest Community (ID: {comm_id}): {stats['size']} nodes**")
                        st.write("Top nodes:")
                        for node in stats['nodes'][:10]:
                            attrs = analytics.node_attributes[node]
                            st.write(f"- {node} (PageRank: {attrs['pagerank']:.4f})")
                
                elif "bridge nodes" in question_lower:
                    bridges = analytics.find_bridge_nodes()
                    if bridges:
                        st.write("**Top Bridge Nodes:**")
                        for bridge in bridges[:10]:
                            st.write(f"- **{bridge['node']}** connects {bridge['bridge_strength']} communities")
                    else:
                        st.write("No bridge nodes found.")
                
                elif "hub nodes" in question_lower or "most connected" in question_lower:
                    hubs = analytics.identify_hub_nodes(10)
                    st.write("**Top Hub Nodes:**")
                    for node, score in hubs:
                        st.write(f"- **{node}** (Hub Score: {score:.4f})")
                
                elif "statistics" in question_lower or "stats" in question_lower:
                    st.write("**Network Statistics:**")
                    st.write(f"- Nodes: {len(G.nodes())}")
                    st.write(f"- Edges: {len(G.edges())}")
                    st.write(f"- Average Degree: {np.mean([G.degree(n) for n in G.nodes()]):.2f}")
                    st.write(f"- Density: {nx.density(G):.4f}")
                    st.write(f"- Communities: {len(set(partition.values())) if partition else 'N/A'}")
                    
                    # Connectedness
                    if nx.is_connected(G):
                        st.write(f"- Average Path Length: {nx.average_shortest_path_length(G):.2f}")
                    else:
                        st.write("- Graph is not fully connected")
                        st.write(f"- Connected Components: {nx.number_connected_components(G)}")
                
                else:
                    # Fallback to original chatbot logic
                    st.info("Try asking about: targets, communities, hubs, bridges, or statistics")
        
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
