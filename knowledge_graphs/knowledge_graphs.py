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

import string
st.set_page_config(page_title="Interactive Knowledge Graphs", layout="wide")


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

# File uploader for mapping file
uploaded_file = st.file_uploader("Upload an excel file containing relationships. ", type=["xlsx", "xls"])

# Function to generate the graph
def generate_graph(data, color_by_community, size_by_centrality):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['head'], row['tail'], label=row['relation'])

    # Make the network larger
    net = Network(height="850px", width="100%", notebook=False)

    # Apply Louvain Community Coloring with more colors
    if color_by_community:
        partition = community_louvain.best_partition(G,resolution=1.3,  random_state=42)
        num_communities = len(set(partition.values()))
        
        # Generate appropriate number of colors
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
        node_color = community_colors[partition[node]] if color_by_community else None
        node_size = centrality[node] * 150 if centrality else 25  # Adjust node size scaling
        net.add_node(
            node,
            label=str(node),
            title=f"Node: {node}" + (f", Community: {partition[node]}" if color_by_community else ""),
            color=node_color,
            size=node_size,
            font={"size": 20}
        )

    # Add edges to the graph
    for edge in G.edges(data=True):
        label = edge[2].get('label', '')
        net.add_edge(edge[0], edge[1], title=label, label=label)

    # Configure graph physics
    net.repulsion(node_distance=120, central_gravity=0.2, spring_length=200, spring_strength=0.01)

    return G, net, partition

# Function to create community subgraph visualization with 1-degree expansion
def create_community_subgraph(G, partition, community_id, community_colors=None, expand_one_degree=False):
    # Get nodes in the specific community
    community_nodes = [node for node, comm in partition.items() if comm == community_id]
    
    # If expand_one_degree is True, add neighbors of community nodes
    if expand_one_degree:
        expanded_nodes = set(community_nodes)
        for node in community_nodes:
            neighbors = list(G.neighbors(node))
            expanded_nodes.update(neighbors)
        subgraph_nodes = list(expanded_nodes)
    else:
        subgraph_nodes = community_nodes
    
    # Create subgraph
    subgraph = G.subgraph(subgraph_nodes)
    
    # Create PyVis network
    net = Network(height="600px", width="100%", notebook=False)
    
    # Add nodes with different colors for original community vs expanded nodes
    for node in subgraph.nodes():
        if node in community_nodes:
            # Original community nodes - use community color
            color = community_colors[community_id] if community_colors else "#97c2fc"
            node_title = f"Node: {node}, Community: {community_id}"
        else:
            # Expanded nodes (1-degree neighbors) - use gray
            color = "#d3d3d3"
            node_title = f"Node: {node}, Neighbor of Community {community_id}"
        
        net.add_node(
            node,
            label=str(node),
            title=node_title,
            color=color,
            size=30,
            font={"size": 16}
        )
    
    # Add edges
    for edge in subgraph.edges(data=True):
        label = edge[2].get('label', '')
        net.add_edge(edge[0], edge[1], title=label, label=label)
    
    # Configure physics
    net.repulsion(node_distance=100, central_gravity=0.3, spring_length=150, spring_strength=0.02)
    
    return net, subgraph, subgraph_nodes

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
            partition = community_louvain.best_partition(G, resolution=1.3,  random_state=42)
            num_communities = len(set(partition.values()))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Nodes", num_nodes)
        with col2:
            st.metric("Number of Edges", num_edges)
        with col3:
            st.metric("Number of Communities", num_communities)

        # TOP HUBS AND BRIDGES TABLES
        st.subheader("Network Analysis")
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Create DataFrames for top hubs and bridges
        hubs_df = pd.DataFrame([
            {"Node": node, "Degree Centrality": centrality} 
            for node, centrality in degree_centrality.items()
        ]).sort_values("Degree Centrality", ascending=False).head(15).reset_index(drop=True)
        
        bridges_df = pd.DataFrame([
            {"Node": node, "Betweenness Centrality": centrality} 
            for node, centrality in betweenness_centrality.items()
        ]).sort_values("Betweenness Centrality", ascending=False).head(15).reset_index(drop=True)
        
        # Display tables side by side
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top 15 Hubs (by Degree Centrality)**")
            st.dataframe(hubs_df, use_container_width=True)
        
        with col2:
            st.write("**Top 15 Bridges (by Betweenness Centrality)**")
            st.dataframe(bridges_df, use_container_width=True)

        # NEW: Community Analysis Table (only when color_by_community is enabled)
        if color_by_community:
            st.subheader("Community Analysis")
            
            # Calculate PageRank for all nodes
            pagerank_scores = nx.pagerank(G)
            
            # Find the most central node in each community
            community_analysis = []
            for community_id in sorted(set(partition.values())):
                # Get all nodes in this community
                community_nodes = [node for node, comm in partition.items() if comm == community_id]
                
                # Find the node with highest PageRank in this community
                most_central_node = max(community_nodes, key=lambda node: pagerank_scores[node])
                
                community_analysis.append({
                    "Community": community_id,
                    "Size": len(community_nodes),
                    "Central Node": most_central_node,
                    "PageRank Score": round(pagerank_scores[most_central_node], 4),
                    "Degree": G.degree(most_central_node)
                })
            
            # Create and display the community analysis table
            community_df = pd.DataFrame(community_analysis)
            st.write("**Most Central Node in Each Community (by PageRank)**")
            st.dataframe(community_df, use_container_width=True)
            
            st.caption("💡 **Tip**: These central nodes are good starting points for exploring each community. Click on a community number in your chatbot to visualize it!")

        # Display the graph
        st.subheader("Knowledge Graph Visualization")
        components.html(st.session_state["graph_html"], height=1200, scrolling=False)

        # Chatbot Interface
        st.subheader("Ask Questions About the Knowledge Graph")
        user_question = st.text_input("Ask a question (e.g., 'Which node has the most connections?', 'How many modules are there?','Who are the tryptophan neighbors?', 'Connection between B. infantis and stress scores', 'Who else is in the B. infantis module?' (with optional 1-degree expansion), 'Show me the community insulin is a part of', 'Show me community 0'): ")

        if user_question:
            # Analyze user question using SpaCy
            doc = nlp(user_question)
            entities = [ent.text for ent in doc.ents]
            keywords = [token.text.lower() for token in doc if not token.is_stop]

            # General Queries
            if "most connections" in user_question.lower():
                most_connected_node = max(G.degree, key=lambda x: x[1])
                st.write(f"The node with the most connections is **{most_connected_node[0]}** with **{most_connected_node[1]} connections**.")

            elif "how many modules" in user_question.lower():
                num_modules = len(set(partition.values()))
                st.write(f"There are **{num_modules} modules** in the graph.")

            # NEW: Show community that a specific node is part of
            elif "show me the community" in user_question.lower() and "is a part of" in user_question.lower():
                # Extract node name from query
                cleaned_query = preprocess_query(user_question)
                words = cleaned_query.split()
                start_idx = words.index("community") + 1
                end_idx = words.index("is")
                node_name = " ".join(words[start_idx:end_idx])
                
                # Fuzzy match the node name
                best_match, score = process.extractOne(node_name, [node.lower() for node in G.nodes()])
                
                if best_match and score > 70:
                    matched_node = [node for node in G.nodes() if node.lower() == best_match][0]
                    community_id = partition[matched_node]
                    community_nodes = [node for node, comm in partition.items() if comm == community_id]
                    
                    st.write(f"**Node '{matched_node}' is part of Community {community_id}**")
                    st.write(f"**Community {community_id} contains {len(community_nodes)} nodes:**")
                    
                    # Create community colors for visualization
                    colors = plt.cm.tab10(range(len(set(partition.values()))))
                    community_colors = {community: rgb2hex(color[:3]) for community, color in enumerate(colors)}
                    
                    # Create and display community subgraph
                    community_net, subgraph, _ = create_community_subgraph(G, partition, community_id, community_colors)
                    community_net.save_graph("community_graph.html")
                    
                    with open("community_graph.html", 'r') as f:
                        community_html = f.read()
                    
                    st.write("**Community Visualization:**")
                    components.html(community_html, height=700, scrolling=False)
                    
                    # List all nodes in the community
                    st.write("**All nodes in this community:**")
                    for i, node in enumerate(community_nodes, 1):
                        st.write(f"{i}. {node}")
                
                else:
                    st.write(f"No close match found for '{node_name}'. Please check your query.")

            # NEW: Show specific community by number
            elif "show me community" in user_question.lower() and user_question.lower().split()[-1].isdigit():
                community_id = int(user_question.lower().split()[-1])
                
                if community_id in set(partition.values()):
                    community_nodes = [node for node, comm in partition.items() if comm == community_id]
                    
                    st.write(f"**Community {community_id} contains {len(community_nodes)} nodes:**")
                    
                    # Create community colors for visualization
                    colors = plt.cm.tab10(range(len(set(partition.values()))))
                    community_colors = {community: rgb2hex(color[:3]) for community, color in enumerate(colors)}
                    
                    # Create and display community subgraph
                    community_net, subgraph, _ = create_community_subgraph(G, partition, community_id, community_colors)
                    community_net.save_graph("community_graph.html")
                    
                    with open("community_graph.html", 'r') as f:
                        community_html = f.read()
                    
                    st.write("**Community Visualization:**")
                    components.html(community_html, height=700, scrolling=False)
                    
                    # List all nodes in the community
                    st.write("**All nodes in this community:**")
                    for i, node in enumerate(community_nodes, 1):
                        st.write(f"{i}. {node}")
                
                else:
                    available_communities = sorted(set(partition.values()))
                    st.write(f"Community {community_id} does not exist. Available communities are: {available_communities}")

            # Updated "who else is in the module" with 1-degree expansion
            elif "who else is in the" in preprocess_query(user_question) and "module" in preprocess_query(user_question):
                # Normalize nodes for case-insensitive matching
                normalized_nodes = {node.lower(): node for node in G.nodes()}

                # Extract the potential module name from the query
                cleaned_query = preprocess_query(user_question)
                words = cleaned_query.split()
                potential_module_name = " ".join(words[words.index("the") + 1 : words.index("module")])

                # Fuzzy match the module name to node names
                best_match, score = process.extractOne(potential_module_name, list(normalized_nodes.keys()))

                if best_match and score > 70:
                    node_name = normalized_nodes[best_match]
                    if node_name in G.nodes():
                        # Identify the module and list all nodes in it
                        node_module = partition[node_name]
                        same_module_nodes = [n for n, mod in partition.items() if mod == node_module]
                        
                        st.write(f"**Node '{node_name}' is in module {node_module}, which contains {len(same_module_nodes)} nodes:**")
                        
                        # Add checkbox for 1-degree expansion
                        expand_module = st.checkbox(
                            f"🔍 Show 1-degree neighbors of module {node_module}", 
                            key=f"expand_module_{node_module}",
                            help="Include nodes that are directly connected to any node in this module"
                        )
                        
                        # Create community colors for visualization
                        num_communities = len(set(partition.values()))
                        if num_communities <= 10:
                            colors = plt.cm.tab10(range(num_communities))
                        elif num_communities <= 20:
                            colors = plt.cm.tab20(range(num_communities))
                        else:
                            colors = plt.cm.hsv(np.linspace(0, 1, num_communities))
                        community_colors = {community: rgb2hex(color[:3]) for community, color in enumerate(colors)}
                        
                        # Create and display module subgraph
                        module_net, subgraph, displayed_nodes = create_community_subgraph(
                            G, partition, node_module, community_colors, expand_module
                        )
                        module_net.save_graph("module_graph.html")
                        
                        with open("module_graph.html", 'r') as f:
                            module_html = f.read()
                        
                        expansion_text = " + 1-degree neighbors" if expand_module else ""
                        st.write(f"**Module {node_module} Visualization{expansion_text}:**")
                        components.html(module_html, height=700, scrolling=False)
                        
                        # Show statistics and node lists
                        if expand_module:
                            neighbor_nodes = [node for node in displayed_nodes if node not in same_module_nodes]
                            st.write(f"**Module {node_module} has {len(same_module_nodes)} core nodes + {len(neighbor_nodes)} neighbors = {len(displayed_nodes)} total nodes displayed**")
                            
                            # List core module nodes
                            st.write("**Core module nodes:**")
                            for i, n in enumerate(same_module_nodes, 1):
                                st.write(f"{i}. {n}")
                            
                            # List neighbor nodes
                            if neighbor_nodes:
                                st.write("**1-degree neighbors:**")
                                for i, node in enumerate(neighbor_nodes, 1):
                                    # Show which module each neighbor belongs to
                                    neighbor_module = partition.get(node, "Unknown")
                                    st.write(f"{i}. {node} (Module {neighbor_module})")
                        else:
                            st.write("**All nodes in this module:**")
                            for i, n in enumerate(same_module_nodes, 1):
                                st.write(f"{i}. {n}")
                    else:
                        st.write(f"No node matched '{user_question}'. Please try again.")
                else:
                    st.write(f"No close match found for '{potential_module_name}'. Please check your query.")

            elif "connection between" in user_question.lower():
                # Extract the two nodes from the query
                cleaned_query = preprocess_query(user_question)
                if " and " in cleaned_query:
                    node1, node2 = map(str.strip, cleaned_query.split(" and "))
                    node1 = process.extractOne(node1, [n.lower() for n in G.nodes()])[0]  # Fuzzy match node1
                    node2 = process.extractOne(node2, [n.lower() for n in G.nodes()])[0]  # Fuzzy match node2

                    # Get original case-sensitive node names
                    node1 = [n for n in G.nodes() if n.lower() == node1][0]
                    node2 = [n for n in G.nodes() if n.lower() == node2][0]

                    if node1 in G.nodes() and node2 in G.nodes():
                        try:
                            # Find the shortest path
                            shortest_path = nx.shortest_path(G, source=node1, target=node2)
                            st.write(f"**Shortest path between '{node1}' and '{node2}':**")
                            for i in range(len(shortest_path) - 1):
                                edge_label = G.edges[shortest_path[i], shortest_path[i + 1]].get("label", "unknown")
                                st.write(f"{shortest_path[i]} -> {shortest_path[i + 1]} via '{edge_label}'")

                            # Highlight the shortest path in PyVis visualization
                            net = Network(height="850px", width="100%", notebook=False)
                            for node in G.nodes():
                                net.add_node(node, label=node, title=f"Node: {node}", color="#97c2fc", size=25)

                            for edge in G.edges(data=True):
                                edge_color = "red" if edge[0] in shortest_path and edge[1] in shortest_path else "gray"
                                edge_width = 5 if edge_color == "red" else 1
                                net.add_edge(
                                    edge[0], edge[1],
                                    title=edge[2].get("label", ""),
                                    label=edge[2].get("label", ""),
                                    color=edge_color,
                                    width=edge_width
                                )

                            # Save and display the updated graph
                            net.save_graph("highlighted_graph.html")
                            components.html(open("highlighted_graph.html", 'r').read(), height=1100, scrolling=False)

                            # Optionally, show additional paths up to a certain length
                            paths = list(nx.all_simple_paths(G, source=node1, target=node2, cutoff=3))
                            st.write(f"**All paths up to length 3 between '{node1}' and '{node2}':**")
                            for path in paths:
                                st.write(" -> ".join(path))

                        except nx.NetworkXNoPath:
                            st.write(f"There is no path between '{node1}' and '{node2}' in the graph.")
                    else:
                        st.write("One or both nodes were not found in the graph.")
                else:
                    st.write("Please format your query as: 'What is the connection between Node1 and Node2?'")

            elif "neighbors" in user_question.lower():
                # Preprocess query to extract the main node
                cleaned_query = preprocess_query(user_question)
                keywords = cleaned_query.split()

                # Assume the main node is the keyword before "neighbors"
                potential_node_name = " ".join(keywords[:keywords.index("neighbors")])

                # Fuzzy match the node name to graph nodes
                best_match, score = process.extractOne(potential_node_name, [node.lower() for node in G.nodes()])

                if best_match and score > 70:
                    matched_node = [node for node in G.nodes() if node.lower() == best_match][0]
                    neighbors = list(G.neighbors(matched_node))

                    if neighbors:
                        # Filter neighbors
                        product_neighbors = neighbors

                        if product_neighbors:
                            # Calculate PageRank for neighbors
                            page_rank = nx.pagerank(G)

                            # Sort product neighbors by PageRank
                            sorted_products = sorted(product_neighbors, key=lambda n: page_rank[n], reverse=True)

                            # Display top 5 products
                            for product in sorted_products[:5]:
                                st.write(f"- **{product.strip()}** (PageRank: {page_rank[product]:.4f})")
                        else:
                            st.write(f"No neighbors directly connected to **{matched_node}**.")
                    else:
                        st.write(f"Node '{matched_node}' has no neighbors.")
                else:
                    st.write(f"No close match found for '{potential_node_name}'. Please check your query.")

    else:
        st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
