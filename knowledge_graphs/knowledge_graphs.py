from fuzzywuzzy import process
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import seaborn as sns
import community.community_louvain as community_louvain
import spacy
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
import string
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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
        
        st.header("🔍 Analysis Focus")
        analysis_mode = st.selectbox(
            "Choose analysis type:",
            ["Overview Dashboard", "Target Discovery", "Community Analysis", "Relationship Patterns", "Interactive Chat"]
        )

# Function to generate the graph (enhanced)
@st.cache_data
def generate_graph(data, color_by_community, size_by_centrality):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['head'], row['tail'], label=row['relation'])

    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "repulsion": {"nodeDistance": 100, "centralGravity": 0.2},
        "solver": "repulsion"
      }
    }
    """)

    # Apply Louvain Community Coloring
    if color_by_community:
        partition = community_louvain.best_partition(G)
        num_communities = len(set(partition.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, num_communities))
        community_colors = {community: rgb2hex(color[:3]) for community, color in enumerate(colors)}
    else:
        partition = None
        community_colors = None

    # Calculate centrality
    centrality_map = {
        "Degree Centrality": nx.degree_centrality(G),
        "Betweenness Centrality": nx.betweenness_centrality(G),
        "PageRank": nx.pagerank(G)
    }
    centrality = centrality_map.get(size_by_centrality)

    # Add nodes to the graph
    for node in G.nodes():
        node_color = community_colors[partition[node]] if color_by_community else "#97c2fc"
        node_size = (centrality[node] * 100 + 10) if centrality else 25
        
        title = f"Node: {node}"
        if color_by_community and partition:
            title += f"<br>Community: {partition[node]}"
        if centrality:
            title += f"<br>{size_by_centrality}: {centrality[node]:.3f}"
        
        net.add_node(node, label=str(node), title=title, color=node_color, size=node_size)

    # Add edges
    for edge in G.edges(data=True):
        label = edge[2].get('label', '')
        net.add_edge(edge[0], edge[1], title=label, label=label)

    return G, net, partition

# Main application logic
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    required_columns = ['head', 'tail', 'relation']
    
    if all(col in data.columns for col in required_columns):
        # Generate graph
        G, net, partition = generate_graph(data, color_by_community, size_by_centrality)
        analytics = KnowledgeGraphAnalytics(G, data, partition)
        
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
                
                # Hub nodes bar chart
                fig = px.bar(hub_df, x='Hub Score', y='Node', orientation='h',
                            title="Hub Nodes by Composite Score")
                st.plotly_chart(fig, use_container_width=True)
            
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
            
            # Network visualization
            st.subheader("🕸️ Interactive Network Visualization")
            net.save_graph("temp_graph.html")
            with open("temp_graph.html", 'r') as f:
                graph_html = f.read()
            components.html(graph_html, height=650)
            
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
                        
                        # Visualize top targets
                        top_targets = targets[:10]
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=[t[1]['pagerank'] for t in top_targets],
                            y=[t[1]['degree'] for t in top_targets],
                            mode='markers+text',
                            text=[t[0][:20] + '...' if len(t[0]) > 20 else t[0] for t in top_targets],
                            textposition="top center",
                            marker=dict(
                                size=[t[1]['betweenness'] * 1000 + 10 for t in top_targets],
                                color=[t[1]['community'] for t in top_targets],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Community")
                            ),
                            name="Potential Targets"
                        ))
                        
                        fig.update_layout(
                            title="Potential Targets: PageRank vs Degree (Size = Betweenness)",
                            xaxis_title="PageRank Score",
                            yaxis_title="Node Degree",
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
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
                    fig = px.scatter(comm_df, x='Size', y='Modularity', 
                                   hover_data=['Community', 'Avg PageRank'],
                                   title="Community Size vs Modularity")
                    st.plotly_chart(fig, use_container_width=True)
                
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
                        fig = px.bar(rel_df, x='Count', y='Relationship', orientation='h')
                        st.plotly_chart(fig, use_container_width=True)
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
