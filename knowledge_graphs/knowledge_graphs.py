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

import string

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

    net = Network(height="850px", width="100%", notebook=False)

    # Apply Louvain Community Coloring if selected
    if color_by_community:
        partition = community_louvain.best_partition(G)
        num_communities = len(set(partition.values()))
        colors = plt.cm.tab10(range(num_communities))
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

        # Display the graph
        components.html(st.session_state["graph_html"], height=1100, width=1300)

        # Chatbot Interface
        st.subheader("Ask Questions About the Knowledge Graph")
        user_question = st.text_input("Ask a question (e.g., 'Which node has the most connections?', 'How many modules are there?','Who are the tryptophan neighbors?', 'Connection between B. infantis and stress scores',or 'Who else is in the B. infantis module?'): ")

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



            elif "who else is in the" in preprocess_query(user_question) and "module" in preprocess_query(user_question):
                # Normalize nodes for case-insensitive matching
                normalized_nodes = {node.lower(): node for node in G.nodes()}  # Lowercase graph node names

                # Extract the potential module name from the query
                cleaned_query = preprocess_query(user_question)
                words = cleaned_query.split()
                potential_module_name = " ".join(words[words.index("the") + 1 : words.index("module")])

                # Fuzzy match the module name to node names
                best_match, score = process.extractOne(potential_module_name, list(normalized_nodes.keys()))

                if best_match and score > 70:  # Adjust threshold if needed
                    node_name = normalized_nodes[best_match]  # Get the original node name
                    if node_name in G.nodes():
                        # Identify the module and list all nodes in it
                        node_module = partition[node_name]
                        same_module_nodes = [n for n, mod in partition.items() if mod == node_module]
                        st.write(f"**Node '{node_name}' is in module {node_module}, which contains the following nodes:**")
                        for n in same_module_nodes:
                            st.write(f"- {n}")
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
                                    title=edge[2].get("label", ""),  # Access the "data" part of the edge tuple
                                    label=edge[2].get("label", ""),  # Access the "data" part of the edge tuple
                                    color=edge_color,
                                    width=edge_width
                                )


                            # Save and display the updated graph
                            net.save_graph("highlighted_graph.html")
                            components.html(open("highlighted_graph.html", 'r').read(), height=1100, width=1300)

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
                keywords = cleaned_query.split()  # Split query into words

                # Assume the main node is the keyword before "product"
                potential_node_name = " ".join(keywords[:keywords.index("neighbors")])

                # Fuzzy match the node name to graph nodes
                best_match, score = process.extractOne(potential_node_name, [node.lower() for node in G.nodes()])
                # st.write(f"Best match: {best_match}, Score: {score}")

                if best_match and score > 70:
                    matched_node = [node for node in G.nodes() if node.lower() == best_match][0]  # Get original node name
                    neighbors = list(G.neighbors(matched_node))
                    # st.write(f"Best neighbors: {neighbors}")

                    if neighbors:
                        # Filter neighbors with "product" in edge labels connected to the matched node
                        product_neighbors=neighbors
                        # future can filter based on connecitons beign products or not
                        # product_neighbors = [
                        #     neighbor for neighbor in neighbors
                        #     if "product" in G.edges[matched_node, neighbor].get("label", "").lower()
                        # ]
                        # st.write(f" product_neighbors: {product_neighbors}")

                        if product_neighbors:
                            # Calculate PageRank for neighbors
                            page_rank = nx.pagerank(G)

                            # Sort product neighbors by PageRank
                            sorted_products = sorted(product_neighbors, key=lambda n: page_rank[n], reverse=True)

                            # Display top 5 products
                            # st.write(f"Top 5 products connected to **{matched_node}** based on PageRank:")
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
