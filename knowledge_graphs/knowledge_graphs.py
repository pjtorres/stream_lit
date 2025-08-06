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
        """Comprehensive community analysis with FIXED counting"""
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
        
        # FIXED: Properly count all nodes in each community
        for node in self.G.nodes():
            comm = self.partition[node]
            community_stats[comm]['nodes'].append(node)
        
        # Update sizes after collecting all nodes
        for comm_id, stats in community_stats.items():
            stats['size'] = len(stats['nodes'])  # FIXED: Correct size calculation
        
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

# FIXED: Get community subgraph with controllable expansion - ensures correct community filtering
def get_community_subgraph(G, partition, focus_community, expansion_degree=0):
    """FIXED: Get community subgraph with controllable expansion - ensures correct community filtering"""
    if focus_community is None or partition is None:
        return G, partition
    
    # Start with core community nodes - FIXED to use correct community ID
    core_nodes = {node for node, comm in partition.items() if comm == focus_community}
    
    if len(core_nodes) == 0:
        st.warning(f"No nodes found in community {focus_community}")
        return G, partition
    
    if expansion_degree == 0:
        # Only core community nodes
        selected_nodes = core_nodes
    else:
        # Expand by N degrees
        selected_nodes = set(core_nodes)
        current_frontier = core_nodes
        
        for degree in range(expansion_degree):
            next_frontier = set()
            for node in current_frontier:
                if node in G.nodes():  # Safety check
                    for neighbor in G.neighbors(node):
                        if neighbor not in selected_nodes:
                            next_frontier.add(neighbor)
                            selected_nodes.add(neighbor)
            current_frontier = next_frontier
            
            if not current_frontier:  # No more nodes to expand to
                break
    
    # Create subgraph - FIXED to ensure all selected nodes exist
    valid_nodes = {node for node in selected_nodes if node in G.nodes()}
    if len(valid_nodes) == 0:
        st.warning("No valid nodes found for subgraph")
        return G, partition
        
    G_filtered = G.subgraph(valid_nodes).copy()
    
    # Update partition for filtered graph - FIXED to maintain correct community assignments
    partition_filtered = {node: partition[node] for node in G_filtered.nodes() if node in partition}
    
    return G_filtered, partition_filtered

# Function to generate the graph (FIXED with proper community filtering)
@st.cache_data
def generate_graph(data, color_by_community, size_by_centrality, focus_community=None, 
                   expansion_degree=1, graph_size="large"):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['head'], row['tail'], label=row['relation'])

    # Apply Louvain Community Detection on FULL graph first
    partition = None
    community_colors = None
    
    if color_by_community:
        partition = community_louvain.best_partition(G)
        num_communities = len(set(partition.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, num_communities))
        community_colors = {community: rgb2hex(color[:3]) for community, color in enumerate(colors)}

    # Store original partition for analytics
    original_partition = partition.copy() if partition else None

    # FIXED: Apply community filtering AFTER community detection
    if focus_community is not None and partition:
        # Validate that the focus_community exists
        available_communities = set(partition.values())
        if focus_community not in available_communities:
            st.error(f"Community {focus_community} not found. Available communities: {sorted(available_communities)}")
            focus_community = None
        else:
            G, partition = get_community_subgraph(G, partition, focus_community, expansion_degree)

    # Set visualization size with REDUCED NODE SIZES
    if graph_size == "extra_large":
        height, width = "900px", "100%"
        physics_distance = 150
        node_base_size = 15
    elif graph_size == "large":
        height, width = "750px", "100%"
        physics_distance = 120
        node_base_size = 12
    else:  # medium
        height, width = "600px", "100%"
        physics_distance = 100
        node_base_size = 10

    net = Network(height=height, width=width, bgcolor="#222222", font_color="white")
    
    # Enhanced physics for better layout
    net.set_options(f"""
    var options = {{
      "physics": {{
        "enabled": true,
        "repulsion": {{"nodeDistance": {physics_distance}, "centralGravity": 0.3, "springLength": 200}},
        "solver": "repulsion",
        "stabilization": {{"iterations": 150}}
      }},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 200,
        "zoomView": true,
        "dragView": true
      }},
      "nodes": {{
        "font": {{"size": 14, "color": "white"}},
        "borderWidth": 2
      }},
      "edges": {{
        "font": {{"size": 12, "color": "white", "strokeWidth": 0, "strokeColor": "black"}},
        "smooth": {{"type": "continuous"}}
      }}
    }}
    """)

    # Calculate centrality on the CURRENT graph (filtered or full)
    centrality_map = {
        "Degree Centrality": nx.degree_centrality(G),
        "Betweenness Centrality": nx.betweenness_centrality(G),
        "PageRank": nx.pagerank(G)
    }
    centrality = centrality_map.get(size_by_centrality)

    # Add nodes to the graph with FIXED community highlighting
    for node in G.nodes():
        if color_by_community and partition and node in partition:
            node_community = partition[node]
            node_color = community_colors.get(node_community, "#97c2fc")
            
            # FIXED: Highlight focused community nodes correctly
            if focus_community is not None and node_community == focus_community:
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
        if centrality and node in centrality:
            node_size = (centrality[node] * 50 + node_base_size)
        else:
            node_size = node_base_size
            
        # Make focused community nodes larger
        if focus_community is not None and partition and node in partition and partition[node] == focus_community:
            node_size *= 1.3  # More visible increase
        
        # Create detailed tooltip
        title = f"<b>{node}</b>"
        if color_by_community and partition and node in partition:
            title += f"<br>Community: {partition[node]}"
        title += f"<br>Degree: {G.degree(node)}"
        if centrality and node in centrality:
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
            node1_comm = partition.get(edge[0], -1)
            node2_comm = partition.get(edge[1], -1)
            
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

    # Return original partition for analytics, but filtered graph for visualization
    return G, net, original_partition

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

# Main application logic
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    required_columns = ['head', 'tail', 'relation']
    
    if all(col in data.columns for col in required_columns):
        # Generate initial graph to get community info - FIXED to use original partition
        partition_all = community_louvain.best_partition(nx.from_pandas_edgelist(data, 'head', 'tail'))
        all_G = nx.from_pandas_edgelist(data, 'head', 'tail', edge_attr=True)
        full_analytics = KnowledgeGraphAnalytics(all_G, data, partition_all)
        community_stats_all = full_analytics.community_analysis()
                
        # FIXED Community selection with proper controls
        focus_community = None
        expansion_degree = 1
        
        if color_by_community and partition_temp:
            st.sidebar.header("🎯 Community Focus Controls")
            
            # Get CORRECT community stats
            community_stats = analytics.community_analysis()
            
            # Create community options with CORRECT sizes
            community_options = ["All Communities (Full Graph)"]
            for comm_id, stats in sorted(community_stats_all.items(), key=lambda x: x[1]['size'], reverse=True):
                community_options.append(f"Community {comm_id} ({stats['size']} nodes)")
            selected_community = st.sidebar.selectbox("Focus on specific community:", community_options)

            focus_community = None
            expansion_degree = 1

            if selected_community != "All Communities (Full Graph)":
                focus_community = int(selected_community.split()[1])
                st.sidebar.subheader("🔍 Expansion Control")
                expansion_type = st.sidebar.radio("View mode:", ["Core only", "Expand by degrees"])
                if expansion_type == "Core only":
                    expansion_degree = 0
                else:
                    expansion_degree = st.sidebar.slider("Expansion degrees:", 1, 3, 1)

            # Step 3: Generate the graph ONCE with focus parameters:
            G, net, partition = generate_graph(data, color_by_community, size_by_centrality, 
                                               focus_community, expansion_degree, graph_size)
            analytics = KnowledgeGraphAnalytics(G, data, partition)
        # # Generate final graph with FIXED focus and expansion
        # G, net, partition = generate_graph(data, color_by_community, size_by_centrality, 
        #                                  focus_community, expansion_degree, graph_size)
        
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
            
            # Show focus info with FIXED counts
            if focus_community is not None:
                if partition:
                    core_nodes_in_view = [n for n in G.nodes() if partition.get(n) == focus_community]
                    total_nodes_in_view = len(G.nodes())
                    
                    if expansion_degree == 0:
                        st.subheader(f"🎯 Community {focus_community} - Core Nodes Only")
                        st.info(f"Showing {len(core_nodes_in_view)} core nodes from Community {focus_community}")
                    else:
                        st.subheader(f"🎯 Community {focus_community} + {expansion_degree} Degree Expansion")
                        st.info(f"Showing {len(core_nodes_in_view)} core nodes + {total_nodes_in_view - len(core_nodes_in_view)} expanded nodes (total: {total_nodes_in_view})")
            else:
                st.subheader("🕸️ Complete Network Visualization")
            
            # Add reset button when focused
            if focus_community is not None:
                if st.button("🔄 Return to Full Network View"):
                    st.experimental_rerun()
            
            # Use ORIGINAL partition for analytics to get correct stats
            analytics_for_display = KnowledgeGraphAnalytics(G, data, partition)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Top Hub Nodes")
                hubs = analytics_for_display.identify_hub_nodes(10)
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
                bridges = analytics_for_display.find_bridge_nodes()[:10]
                if bridges:
                    bridge_df = pd.DataFrame([{
                        'Node': b['node'],
                        'Communities Connected': b['bridge_strength'],
                        'Betweenness': f"{b['betweenness']:.3f}"
                    } for b in bridges])
                    st.dataframe(bridge_df, use_container_width=True)
                else:
                    st.info("No bridge nodes found (requires community detection)")
            
            # Network visualization
            net.save_graph("temp_graph.html")
            with open("temp_graph.html", 'r') as f:
                graph_html = f.read()
            
            # Adjust height based on graph size
            height_map = {"medium": 650, "large": 800, "extra_large": 950}
            components.html(graph_html, height=height_map.get(graph_size, 800))
            
            # Relationship analysis
            st.subheader("🔗 Relationship Type Analysis")
            relation_stats = analytics_for_display.analyze_relationship_patterns()
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
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(top_relations['Frequency'], labels=top_relations['Relation'], autopct='%1.1f%%')
                ax.set_title('Top Relationship Types')
                st.pyplot(fig)
                
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
                
                # Use full graph for target discovery
                full_analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
                
                # Fuzzy match seed nodes to actual graph nodes
                matched_seeds = []
                for seed in seed_nodes:
                    matches = process.extract(seed.lower(), 
                                            [n.lower() for n in G_temp.nodes()], 
                                            limit=3)
                    if matches and matches[0][1] > 60:  # 60% similarity threshold
                        original_node = [n for n in G_temp.nodes() if n.lower() == matches[0][0]][0]
                        matched_seeds.append(original_node)
                        st.success(f"Matched '{seed}' to '{original_node}'")
                    else:
                        st.warning(f"No close match found for '{seed}'")
                
                if matched_seeds:
                    targets = full_analytics.find_potential_targets(
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
            st.header("🏘️ Community Analysis")
            
            if partition_temp:
                # Use the ORIGINAL full graph partition for community analysis
                full_analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
                community_stats = full_analytics.community_analysis()
                
                st.subheader("📈 Community Overview")
                
                # Community summary table with FIXED sizes
                summary_data = []
                for comm_id, stats in sorted(community_stats.items(), key=lambda x: x[1]['size'], reverse=True):
                    summary_data.append({
                        'Community': comm_id,
                        'Size': stats['size'],  # Now correctly calculated
                        'Internal Edges': stats['internal_edges'],
                        'External Edges': stats['external_edges'],
                        'Avg Centrality': f"{stats['avg_centrality']:.4f}",
                        'Top Relations': ', '.join([rel for rel, count in stats['key_relations'].most_common(3)])
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Community size distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Community Size Distribution")
                    sizes = [stats['size'] for stats in community_stats.values()]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(sizes, bins=min(10, len(set(sizes))), edgecolor='black')
                    ax.set_xlabel('Community Size')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Community Sizes')
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("🔗 Internal vs External Connections")
                    
                    internal_edges = [stats['internal_edges'] for stats in community_stats.values()]
                    external_edges = [stats['external_edges'] for stats in community_stats.values()]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    communities = list(community_stats.keys())
                    
                    x = np.arange(len(communities))
                    width = 0.35
                    
                    ax.bar(x - width/2, internal_edges, width, label='Internal Edges')
                    ax.bar(x + width/2, external_edges, width, label='External Edges')
                    
                    ax.set_xlabel('Community')
                    ax.set_ylabel('Number of Edges')
                    ax.set_title('Internal vs External Connections by Community')
                    ax.set_xticks(x)
                    ax.set_xticklabels(communities)
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Detailed community analysis
                st.subheader("🔍 Detailed Community Analysis")
                selected_comm = st.selectbox(
                    "Select community for detailed analysis:",
                    options=list(community_stats.keys()),
                    format_func=lambda x: f"Community {x} ({community_stats[x]['size']} nodes)"
                )
                
                if selected_comm is not None:
                    stats = community_stats[selected_comm]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Community Members:**")
                        members_df = pd.DataFrame({
                            'Node': stats['nodes'],
                            'Centrality': [full_analytics.node_attributes[node]['pagerank'] for node in stats['nodes']]
                        }).sort_values('Centrality', ascending=False)
                        st.dataframe(members_df, use_container_width=True)
                    
                    with col2:
                        st.write("**Relationship Types:**")
                        relations_df = pd.DataFrame([
                            {'Relation': rel, 'Count': count}
                            for rel, count in stats['key_relations'].most_common(10)
                        ])
                        st.dataframe(relations_df, use_container_width=True)
                        
                        # Relationship pie chart
                        if len(relations_df) > 0:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.pie(relations_df['Count'], labels=relations_df['Relation'], autopct='%1.1f%%')
                            ax.set_title(f'Relationship Types in Community {selected_comm}')
                            st.pyplot(fig)
            else:
                st.warning("Community detection not available. Enable 'Color by Communities' in the sidebar.")
        
        elif analysis_mode == "Relationship Patterns":
            st.header("🔗 Relationship Pattern Analysis")
            
            # Use full graph analytics for relationship patterns
            full_analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
            relation_stats = full_analytics.analyze_relationship_patterns()
            
            st.subheader("📊 Relationship Statistics")
            
            # Enhanced relationship analysis
            enhanced_stats = []
            for relation, stats in relation_stats.items():
                # Find nodes most associated with this relation
                relation_data = data[data['relation'] == relation]
                all_nodes_in_relation = list(relation_data['head']) + list(relation_data['tail'])
                node_counts = Counter(all_nodes_in_relation)
                top_nodes = node_counts.most_common(3)
                
                enhanced_stats.append({
                    'Relation': relation,
                    'Frequency': stats['frequency'],
                    'Unique Nodes': stats['unique_nodes'],
                    'Density': f"{stats['density']:.3f}",
                    'Top Nodes': ', '.join([f"{node}({count})" for node, count in top_nodes])
                })
            
            enhanced_df = pd.DataFrame(enhanced_stats).sort_values('Frequency', ascending=False)
            st.dataframe(enhanced_df, use_container_width=True)
            
            # Relationship network analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Relationship Frequency")
                
                top_relations = enhanced_df.head(10)
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(top_relations['Relation'], top_relations['Frequency'])
                ax.set_xlabel('Frequency')
                ax.set_title('Most Common Relationship Types')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("🎯 Relationship Density Analysis")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                densities = [float(d) for d in enhanced_df['Density']]
                ax.scatter(enhanced_df['Unique Nodes'], densities, alpha=0.6)
                ax.set_xlabel('Unique Nodes Involved')
                ax.set_ylabel('Relationship Density')
                ax.set_title('Relationship Density vs Node Coverage')
                
                # Add labels for interesting points
                for idx, row in enhanced_df.iterrows():
                    if float(row['Density']) > 0.1 or row['Unique Nodes'] > enhanced_df['Unique Nodes'].mean():
                        ax.annotate(row['Relation'], 
                                  (row['Unique Nodes'], float(row['Density'])),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.7)
                
                st.pyplot(fig)
            
            # Relationship co-occurrence analysis
            st.subheader("🔄 Relationship Co-occurrence")
            
            # Find nodes that participate in multiple relationship types
            node_relations = defaultdict(set)
            for _, row in data.iterrows():
                node_relations[row['head']].add(row['relation'])
                node_relations[row['tail']].add(row['relation'])
            
            multi_relation_nodes = {node: list(relations) 
                                  for node, relations in node_relations.items() 
                                  if len(relations) > 1}
            
            if multi_relation_nodes:
                # Show nodes with most diverse relationships
                diverse_nodes = sorted(multi_relation_nodes.items(), 
                                     key=lambda x: len(x[1]), reverse=True)[:10]
                
                diverse_df = pd.DataFrame([
                    {
                        'Node': node,
                        'Relationship Count': len(relations),
                        'Relationship Types': ', '.join(relations)
                    }
                    for node, relations in diverse_nodes
                ])
                
                st.write("**Nodes with Most Diverse Relationships:**")
                st.dataframe(diverse_df, use_container_width=True)
        
        elif analysis_mode == "Interactive Chat":
            st.header("💬 Interactive Knowledge Graph Chat")
            
            st.markdown("""
            **Ask questions about your knowledge graph and get intelligent answers based on network analysis.**
            """)
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Chat interface
            user_question = st.text_input("Ask a question about your knowledge graph:", 
                                        placeholder="e.g., What are the most important nodes? Which communities are most connected?")
            
            if st.button("🚀 Ask") and user_question:
                # Use full graph for chat analysis
                full_analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
                
                # Simple question processing
                question_lower = user_question.lower()
                response = ""
                
                # Hub/important nodes questions
                if any(keyword in question_lower for keyword in ['important', 'hub', 'central', 'key']):
                    hubs = full_analytics.identify_hub_nodes(5)
                    response = f"The most important/central nodes in your graph are: {', '.join([hub[0] for hub in hubs[:5]])}. These nodes have high centrality scores and are well-connected to other parts of the network."
                
                # Community questions
                elif any(keyword in question_lower for keyword in ['community', 'cluster', 'group']):
                    if partition_temp:
                        community_stats = full_analytics.community_analysis()
                        largest_communities = sorted(community_stats.items(), key=lambda x: x[1]['size'], reverse=True)[:3]
                        response = f"Your graph has {len(community_stats)} communities. The largest communities are: " + \
                                 ", ".join([f"Community {comm} ({stats['size']} nodes)" for comm, stats in largest_communities])
                    else:
                        response = "Community detection is not enabled. Please enable 'Color by Communities' in the sidebar to analyze communities."
                
                # Bridge nodes questions
                elif any(keyword in question_lower for keyword in ['bridge', 'connect', 'between']):
                    bridges = full_analytics.find_bridge_nodes()
                    if bridges:
                        top_bridges = bridges[:3]
                        response = f"The main bridge nodes that connect different communities are: {', '.join([bridge['node'] for bridge in top_bridges])}. These nodes are crucial for information flow between different parts of the network."
                    else:
                        response = "No bridge nodes identified. This might mean your graph is very well connected or community detection needs to be enabled."
                
                # Relationship questions
                elif any(keyword in question_lower for keyword in ['relation', 'connection', 'edge']):
                    relation_stats = full_analytics.analyze_relationship_patterns()
                    top_relations = sorted(relation_stats.items(), key=lambda x: x[1]['frequency'], reverse=True)[:3]
                    response = f"The most common relationships in your graph are: " + \
                             ", ".join([f"{rel} ({stats['frequency']} occurrences)" for rel, stats in top_relations])
                
                # Target/recommendation questions
                elif any(keyword in question_lower for keyword in ['target', 'recommend', 'suggest', 'find']):
                    response = "To find targets or recommendations, please use the 'Target Discovery' mode and specify seed nodes you're interested in. I can then analyze the network to suggest related nodes based on proximity and relationships."
                
                # Size/scale questions
                elif any(keyword in question_lower for keyword in ['size', 'big', 'large', 'how many']):
                    response = f"Your knowledge graph contains {len(G_temp.nodes())} nodes and {len(G_temp.edges())} edges. The average degree (connections per node) is {np.mean([G_temp.degree(n) for n in G_temp.nodes()]):.1f}."
                
                # Default response
                else:
                    response = "I can help you analyze your knowledge graph! Try asking about:\n\n" + \
                             "• Important or central nodes\n" + \
                             "• Communities and clusters\n" + \
                             "• Bridge nodes and connections\n" + \
                             "• Relationship patterns\n" + \
                             "• Graph size and statistics\n\n" + \
                             "Or use the specific analysis modes for more detailed insights."
                
                # Add to chat history
                st.session_state.chat_history.append({"question": user_question, "response": response})
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("💬 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q: {chat['question']}", expanded=(i==0)):
                        st.write(f"**A:** {chat['response']}")
                
                # Clear chat history button
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.experimental_rerun()
                    
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
    st.dataframe(sample_df, use_container_width=True)from fuzzywuzzy import process
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
        """Comprehensive community analysis with FIXED counting"""
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
        
        # FIXED: Properly count all nodes in each community
        for node in self.G.nodes():
            comm = self.partition[node]
            community_stats[comm]['nodes'].append(node)
        
        # Update sizes after collecting all nodes
        for comm_id, stats in community_stats.items():
            stats['size'] = len(stats['nodes'])  # FIXED: Correct size calculation
        
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

# FIXED: Get community subgraph with controllable expansion - ensures correct community filtering
def get_community_subgraph(G, partition, focus_community, expansion_degree=0):
    """FIXED: Get community subgraph with controllable expansion - ensures correct community filtering"""
    if focus_community is None or partition is None:
        return G, partition
    
    # Start with core community nodes - FIXED to use correct community ID
    core_nodes = {node for node, comm in partition.items() if comm == focus_community}
    
    if len(core_nodes) == 0:
        st.warning(f"No nodes found in community {focus_community}")
        return G, partition
    
    if expansion_degree == 0:
        # Only core community nodes
        selected_nodes = core_nodes
    else:
        # Expand by N degrees
        selected_nodes = set(core_nodes)
        current_frontier = core_nodes
        
        for degree in range(expansion_degree):
            next_frontier = set()
            for node in current_frontier:
                if node in G.nodes():  # Safety check
                    for neighbor in G.neighbors(node):
                        if neighbor not in selected_nodes:
                            next_frontier.add(neighbor)
                            selected_nodes.add(neighbor)
            current_frontier = next_frontier
            
            if not current_frontier:  # No more nodes to expand to
                break
    
    # Create subgraph - FIXED to ensure all selected nodes exist
    valid_nodes = {node for node in selected_nodes if node in G.nodes()}
    if len(valid_nodes) == 0:
        st.warning("No valid nodes found for subgraph")
        return G, partition
        
    G_filtered = G.subgraph(valid_nodes).copy()
    
    # Update partition for filtered graph - FIXED to maintain correct community assignments
    partition_filtered = {node: partition[node] for node in G_filtered.nodes() if node in partition}
    
    return G_filtered, partition_filtered

# Function to generate the graph (FIXED with proper community filtering)
@st.cache_data
def generate_graph(data, color_by_community, size_by_centrality, focus_community=None, 
                   expansion_degree=1, graph_size="large"):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['head'], row['tail'], label=row['relation'])

    # Apply Louvain Community Detection on FULL graph first
    partition = None
    community_colors = None
    
    if color_by_community:
        partition = community_louvain.best_partition(G)
        num_communities = len(set(partition.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, num_communities))
        community_colors = {community: rgb2hex(color[:3]) for community, color in enumerate(colors)}

    # Store original partition for analytics
    original_partition = partition.copy() if partition else None

    # FIXED: Apply community filtering AFTER community detection
    if focus_community is not None and partition:
        # Validate that the focus_community exists
        available_communities = set(partition.values())
        if focus_community not in available_communities:
            st.error(f"Community {focus_community} not found. Available communities: {sorted(available_communities)}")
            focus_community = None
        else:
            G, partition = get_community_subgraph(G, partition, focus_community, expansion_degree)

    # Set visualization size with REDUCED NODE SIZES
    if graph_size == "extra_large":
        height, width = "900px", "100%"
        physics_distance = 150
        node_base_size = 15
    elif graph_size == "large":
        height, width = "750px", "100%"
        physics_distance = 120
        node_base_size = 12
    else:  # medium
        height, width = "600px", "100%"
        physics_distance = 100
        node_base_size = 10

    net = Network(height=height, width=width, bgcolor="#222222", font_color="white")
    
    # Enhanced physics for better layout
    net.set_options(f"""
    var options = {{
      "physics": {{
        "enabled": true,
        "repulsion": {{"nodeDistance": {physics_distance}, "centralGravity": 0.3, "springLength": 200}},
        "solver": "repulsion",
        "stabilization": {{"iterations": 150}}
      }},
      "interaction": {{
        "hover": true,
        "tooltipDelay": 200,
        "zoomView": true,
        "dragView": true
      }},
      "nodes": {{
        "font": {{"size": 14, "color": "white"}},
        "borderWidth": 2
      }},
      "edges": {{
        "font": {{"size": 12, "color": "white", "strokeWidth": 0, "strokeColor": "black"}},
        "smooth": {{"type": "continuous"}}
      }}
    }}
    """)

    # Calculate centrality on the CURRENT graph (filtered or full)
    centrality_map = {
        "Degree Centrality": nx.degree_centrality(G),
        "Betweenness Centrality": nx.betweenness_centrality(G),
        "PageRank": nx.pagerank(G)
    }
    centrality = centrality_map.get(size_by_centrality)

    # Add nodes to the graph with FIXED community highlighting
    for node in G.nodes():
        if color_by_community and partition and node in partition:
            node_community = partition[node]
            node_color = community_colors.get(node_community, "#97c2fc")
            
            # FIXED: Highlight focused community nodes correctly
            if focus_community is not None and node_community == focus_community:
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
        if centrality and node in centrality:
            node_size = (centrality[node] * 50 + node_base_size)
        else:
            node_size = node_base_size
            
        # Make focused community nodes larger
        if focus_community is not None and partition and node in partition and partition[node] == focus_community:
            node_size *= 1.3  # More visible increase
        
        # Create detailed tooltip
        title = f"<b>{node}</b>"
        if color_by_community and partition and node in partition:
            title += f"<br>Community: {partition[node]}"
        title += f"<br>Degree: {G.degree(node)}"
        if centrality and node in centrality:
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
            node1_comm = partition.get(edge[0], -1)
            node2_comm = partition.get(edge[1], -1)
            
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

    # Return original partition for analytics, but filtered graph for visualization
    return G, net, original_partition

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

# Main application logic
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    required_columns = ['head', 'tail', 'relation']
    
    if all(col in data.columns for col in required_columns):
        # Generate initial graph to get community info - FIXED to use original partition
        G_temp, _, partition_temp = generate_graph(data, color_by_community, size_by_centrality)
        analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
        
        # FIXED Community selection with proper controls
        focus_community = None
        expansion_degree = 1
        
        if color_by_community and partition_temp:
            st.sidebar.header("🎯 Community Focus Controls")
            
            # Get CORRECT community stats
            community_stats = analytics.community_analysis()
            
            # Create community options with CORRECT sizes
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
                
                # Validate the community exists
                if focus_community not in community_stats:
                    st.sidebar.error(f"Community {focus_community} not found!")
                    focus_community = None
                else:
                    # FIXED: Add expansion degree control
                    st.sidebar.subheader("🔍 Expansion Control")
                    expansion_type = st.sidebar.radio(
                        "View mode:",
                        ["Core only", "Expand by degrees"],
                        help="Core only: Show only nodes in this community\nExpand by degrees: Include neighboring nodes"
                    )
                    
                    if expansion_type == "Core only":
                        expansion_degree = 0
                    else:
                        expansion_degree = st.sidebar.slider(
                            "Expansion degrees:",
                            min_value=1,
                            max_value=3,
                            value=1,
                            help="Number of degrees to expand from core community"
                        )
                    
                    st.sidebar.success(f"Focusing on Community {focus_community}")
                    
                    # Show CORRECT community info
                    stats = community_stats[focus_community]
                    st.sidebar.write(f"**Community {focus_community} Details:**")
                    st.sidebar.write(f"- Core nodes: {stats['size']}")
                    st.sidebar.write(f"- Internal edges: {stats['internal_edges']}")
                    st.sidebar.write(f"- External edges: {stats['external_edges']}")
                    
                    if expansion_degree > 0:
                        st.sidebar.write(f"- Expansion: +{expansion_degree} degree(s)")
        
        # Generate final graph with FIXED focus and expansion
        G, net, partition = generate_graph(data, color_by_community, size_by_centrality, 
                                         focus_community, expansion_degree, graph_size)
        
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
            
            # Show focus info with FIXED counts
            if focus_community is not None:
                if partition:
                    core_nodes_in_view = [n for n in G.nodes() if partition.get(n) == focus_community]
                    total_nodes_in_view = len(G.nodes())
                    
                    if expansion_degree == 0:
                        st.subheader(f"🎯 Community {focus_community} - Core Nodes Only")
                        st.info(f"Showing {len(core_nodes_in_view)} core nodes from Community {focus_community}")
                    else:
                        st.subheader(f"🎯 Community {focus_community} + {expansion_degree} Degree Expansion")
                        st.info(f"Showing {len(core_nodes_in_view)} core nodes + {total_nodes_in_view - len(core_nodes_in_view)} expanded nodes (total: {total_nodes_in_view})")
            else:
                st.subheader("🕸️ Complete Network Visualization")
            
            # Add reset button when focused
            if focus_community is not None:
                if st.button("🔄 Return to Full Network View"):
                    st.experimental_rerun()
            
            # Use ORIGINAL partition for analytics to get correct stats
            analytics_for_display = KnowledgeGraphAnalytics(G, data, partition)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Top Hub Nodes")
                hubs = analytics_for_display.identify_hub_nodes(10)
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
                bridges = analytics_for_display.find_bridge_nodes()[:10]
                if bridges:
                    bridge_df = pd.DataFrame([{
                        'Node': b['node'],
                        'Communities Connected': b['bridge_strength'],
                        'Betweenness': f"{b['betweenness']:.3f}"
                    } for b in bridges])
                    st.dataframe(bridge_df, use_container_width=True)
                else:
                    st.info("No bridge nodes found (requires community detection)")
            
            # Network visualization
            net.save_graph("temp_graph.html")
            with open("temp_graph.html", 'r') as f:
                graph_html = f.read()
            
            # Adjust height based on graph size
            height_map = {"medium": 650, "large": 800, "extra_large": 950}
            components.html(graph_html, height=height_map.get(graph_size, 800))
            
            # Relationship analysis
            st.subheader("🔗 Relationship Type Analysis")
            relation_stats = analytics_for_display.analyze_relationship_patterns()
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
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(top_relations['Frequency'], labels=top_relations['Relation'], autopct='%1.1f%%')
                ax.set_title('Top Relationship Types')
                st.pyplot(fig)
                
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
                
                # Use full graph for target discovery
                full_analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
                
                # Fuzzy match seed nodes to actual graph nodes
                matched_seeds = []
                for seed in seed_nodes:
                    matches = process.extract(seed.lower(), 
                                            [n.lower() for n in G_temp.nodes()], 
                                            limit=3)
                    if matches and matches[0][1] > 60:  # 60% similarity threshold
                        original_node = [n for n in G_temp.nodes() if n.lower() == matches[0][0]][0]
                        matched_seeds.append(original_node)
                        st.success(f"Matched '{seed}' to '{original_node}'")
                    else:
                        st.warning(f"No close match found for '{seed}'")
                
                if matched_seeds:
                    targets = full_analytics.find_potential_targets(
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
            st.header("🏘️ Community Analysis")
            
            if partition_temp:
                # Use the ORIGINAL full graph partition for community analysis
                full_analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
                community_stats = full_analytics.community_analysis()
                
                st.subheader("📈 Community Overview")
                
                # Community summary table with FIXED sizes
                summary_data = []
                for comm_id, stats in sorted(community_stats.items(), key=lambda x: x[1]['size'], reverse=True):
                    summary_data.append({
                        'Community': comm_id,
                        'Size': stats['size'],  # Now correctly calculated
                        'Internal Edges': stats['internal_edges'],
                        'External Edges': stats['external_edges'],
                        'Avg Centrality': f"{stats['avg_centrality']:.4f}",
                        'Top Relations': ', '.join([rel for rel, count in stats['key_relations'].most_common(3)])
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Community size distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Community Size Distribution")
                    sizes = [stats['size'] for stats in community_stats.values()]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(sizes, bins=min(10, len(set(sizes))), edgecolor='black')
                    ax.set_xlabel('Community Size')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Community Sizes')
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("🔗 Internal vs External Connections")
                    
                    internal_edges = [stats['internal_edges'] for stats in community_stats.values()]
                    external_edges = [stats['external_edges'] for stats in community_stats.values()]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    communities = list(community_stats.keys())
                    
                    x = np.arange(len(communities))
                    width = 0.35
                    
                    ax.bar(x - width/2, internal_edges, width, label='Internal Edges')
                    ax.bar(x + width/2, external_edges, width, label='External Edges')
                    
                    ax.set_xlabel('Community')
                    ax.set_ylabel('Number of Edges')
                    ax.set_title('Internal vs External Connections by Community')
                    ax.set_xticks(x)
                    ax.set_xticklabels(communities)
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Detailed community analysis
                st.subheader("🔍 Detailed Community Analysis")
                selected_comm = st.selectbox(
                    "Select community for detailed analysis:",
                    options=list(community_stats.keys()),
                    format_func=lambda x: f"Community {x} ({community_stats[x]['size']} nodes)"
                )
                
                if selected_comm is not None:
                    stats = community_stats[selected_comm]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Community Members:**")
                        members_df = pd.DataFrame({
                            'Node': stats['nodes'],
                            'Centrality': [full_analytics.node_attributes[node]['pagerank'] for node in stats['nodes']]
                        }).sort_values('Centrality', ascending=False)
                        st.dataframe(members_df, use_container_width=True)
                    
                    with col2:
                        st.write("**Relationship Types:**")
                        relations_df = pd.DataFrame([
                            {'Relation': rel, 'Count': count}
                            for rel, count in stats['key_relations'].most_common(10)
                        ])
                        st.dataframe(relations_df, use_container_width=True)
                        
                        # Relationship pie chart
                        if len(relations_df) > 0:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.pie(relations_df['Count'], labels=relations_df['Relation'], autopct='%1.1f%%')
                            ax.set_title(f'Relationship Types in Community {selected_comm}')
                            st.pyplot(fig)
            else:
                st.warning("Community detection not available. Enable 'Color by Communities' in the sidebar.")
        
        elif analysis_mode == "Relationship Patterns":
            st.header("🔗 Relationship Pattern Analysis")
            
            # Use full graph analytics for relationship patterns
            full_analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
            relation_stats = full_analytics.analyze_relationship_patterns()
            
            st.subheader("📊 Relationship Statistics")
            
            # Enhanced relationship analysis
            enhanced_stats = []
            for relation, stats in relation_stats.items():
                # Find nodes most associated with this relation
                relation_data = data[data['relation'] == relation]
                all_nodes_in_relation = list(relation_data['head']) + list(relation_data['tail'])
                node_counts = Counter(all_nodes_in_relation)
                top_nodes = node_counts.most_common(3)
                
                enhanced_stats.append({
                    'Relation': relation,
                    'Frequency': stats['frequency'],
                    'Unique Nodes': stats['unique_nodes'],
                    'Density': f"{stats['density']:.3f}",
                    'Top Nodes': ', '.join([f"{node}({count})" for node, count in top_nodes])
                })
            
            enhanced_df = pd.DataFrame(enhanced_stats).sort_values('Frequency', ascending=False)
            st.dataframe(enhanced_df, use_container_width=True)
            
            # Relationship network analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Relationship Frequency")
                
                top_relations = enhanced_df.head(10)
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(top_relations['Relation'], top_relations['Frequency'])
                ax.set_xlabel('Frequency')
                ax.set_title('Most Common Relationship Types')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("🎯 Relationship Density Analysis")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                densities = [float(d) for d in enhanced_df['Density']]
                ax.scatter(enhanced_df['Unique Nodes'], densities, alpha=0.6)
                ax.set_xlabel('Unique Nodes Involved')
                ax.set_ylabel('Relationship Density')
                ax.set_title('Relationship Density vs Node Coverage')
                
                # Add labels for interesting points
                for idx, row in enhanced_df.iterrows():
                    if float(row['Density']) > 0.1 or row['Unique Nodes'] > enhanced_df['Unique Nodes'].mean():
                        ax.annotate(row['Relation'], 
                                  (row['Unique Nodes'], float(row['Density'])),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.7)
                
                st.pyplot(fig)
            
            # Relationship co-occurrence analysis
            st.subheader("🔄 Relationship Co-occurrence")
            
            # Find nodes that participate in multiple relationship types
            node_relations = defaultdict(set)
            for _, row in data.iterrows():
                node_relations[row['head']].add(row['relation'])
                node_relations[row['tail']].add(row['relation'])
            
            multi_relation_nodes = {node: list(relations) 
                                  for node, relations in node_relations.items() 
                                  if len(relations) > 1}
            
            if multi_relation_nodes:
                # Show nodes with most diverse relationships
                diverse_nodes = sorted(multi_relation_nodes.items(), 
                                     key=lambda x: len(x[1]), reverse=True)[:10]
                
                diverse_df = pd.DataFrame([
                    {
                        'Node': node,
                        'Relationship Count': len(relations),
                        'Relationship Types': ', '.join(relations)
                    }
                    for node, relations in diverse_nodes
                ])
                
                st.write("**Nodes with Most Diverse Relationships:**")
                st.dataframe(diverse_df, use_container_width=True)
        
        elif analysis_mode == "Interactive Chat":
            st.header("💬 Interactive Knowledge Graph Chat")
            
            st.markdown("""
            **Ask questions about your knowledge graph and get intelligent answers based on network analysis.**
            """)
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Chat interface
            user_question = st.text_input("Ask a question about your knowledge graph:", 
                                        placeholder="e.g., What are the most important nodes? Which communities are most connected?")
            
            if st.button("🚀 Ask") and user_question:
                # Use full graph for chat analysis
                full_analytics = KnowledgeGraphAnalytics(G_temp, data, partition_temp)
                
                # Simple question processing
                question_lower = user_question.lower()
                response = ""
                
                # Hub/important nodes questions
                if any(keyword in question_lower for keyword in ['important', 'hub', 'central', 'key']):
                    hubs = full_analytics.identify_hub_nodes(5)
                    response = f"The most important/central nodes in your graph are: {', '.join([hub[0] for hub in hubs[:5]])}. These nodes have high centrality scores and are well-connected to other parts of the network."
                
                # Community questions
                elif any(keyword in question_lower for keyword in ['community', 'cluster', 'group']):
                    if partition_temp:
                        community_stats = full_analytics.community_analysis()
                        largest_communities = sorted(community_stats.items(), key=lambda x: x[1]['size'], reverse=True)[:3]
                        response = f"Your graph has {len(community_stats)} communities. The largest communities are: " + \
                                 ", ".join([f"Community {comm} ({stats['size']} nodes)" for comm, stats in largest_communities])
                    else:
                        response = "Community detection is not enabled. Please enable 'Color by Communities' in the sidebar to analyze communities."
                
                # Bridge nodes questions
                elif any(keyword in question_lower for keyword in ['bridge', 'connect', 'between']):
                    bridges = full_analytics.find_bridge_nodes()
                    if bridges:
                        top_bridges = bridges[:3]
                        response = f"The main bridge nodes that connect different communities are: {', '.join([bridge['node'] for bridge in top_bridges])}. These nodes are crucial for information flow between different parts of the network."
                    else:
                        response = "No bridge nodes identified. This might mean your graph is very well connected or community detection needs to be enabled."
                
                # Relationship questions
                elif any(keyword in question_lower for keyword in ['relation', 'connection', 'edge']):
                    relation_stats = full_analytics.analyze_relationship_patterns()
                    top_relations = sorted(relation_stats.items(), key=lambda x: x[1]['frequency'], reverse=True)[:3]
                    response = f"The most common relationships in your graph are: " + \
                             ", ".join([f"{rel} ({stats['frequency']} occurrences)" for rel, stats in top_relations])
                
                # Target/recommendation questions
                elif any(keyword in question_lower for keyword in ['target', 'recommend', 'suggest', 'find']):
                    response = "To find targets or recommendations, please use the 'Target Discovery' mode and specify seed nodes you're interested in. I can then analyze the network to suggest related nodes based on proximity and relationships."
                
                # Size/scale questions
                elif any(keyword in question_lower for keyword in ['size', 'big', 'large', 'how many']):
                    response = f"Your knowledge graph contains {len(G_temp.nodes())} nodes and {len(G_temp.edges())} edges. The average degree (connections per node) is {np.mean([G_temp.degree(n) for n in G_temp.nodes()]):.1f}."
                
                # Default response
                else:
                    response = "I can help you analyze your knowledge graph! Try asking about:\n\n" + \
                             "• Important or central nodes\n" + \
                             "• Communities and clusters\n" + \
                             "• Bridge nodes and connections\n" + \
                             "• Relationship patterns\n" + \
                             "• Graph size and statistics\n\n" + \
                             "Or use the specific analysis modes for more detailed insights."
                
                # Add to chat history
                st.session_state.chat_history.append({"question": user_question, "response": response})
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("💬 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q: {chat['question']}", expanded=(i==0)):
                        st.write(f"**A:** {chat['response']}")
                
                # Clear chat history button
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.experimental_rerun()
                    
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
