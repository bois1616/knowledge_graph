# -*- coding: UTF-8 -*-

# import
import pandas as pd
import numpy as np
import os
from pprint import pprint

# from langchain_community.document_loaders import PyPDFLoader,
# UnstructuredPDFLoader,
# PyPDFium2Loader,
# PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

import networkx as nx
import seaborn as sns
from pyvis.network import Network

from helpers.df_helpers import documents_to_dataframe
# This function uses the helpers/prompt function to extract concepts from text
from helpers.df_helpers import dataframe_to_graph
from helpers.df_helpers import graph_to_dataframe
import random

# globals

# functions


def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    # Melt the dataframe into a list of nodes
    dfg_long = pd.melt(df,
                       id_vars=["chunk_id"],
                       value_vars=["node_1", "node_2", ],
                       value_name="node",
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long,
                        on="chunk_id",
                        suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2


def colors_to_community(communities) -> pd.DataFrame:
    ## Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors


if __name__ == '__main__':

    ## Input data directory
    top_dir: Path = Path("..")
    input_directory: Path = top_dir / 'data_input' / 'HotG'

    ## This is where the output csv files will be written
    out_dir: Path = top_dir / 'data_output'
    if not out_dir.exists():
        out_dir.mkdir()

    # File Loader
    loader = DirectoryLoader(input_directory,
                             show_progress=True,
                             )
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len,
            separators=["\n", "\n\n", ],
            is_separator_regex=False,
    )
    pages = loader.load_and_split(text_splitter=splitter)
    print("Number of chunks = ", len(pages))
    print(pages[3].page_content)

    df = documents_to_dataframe(pages)
    print(df.shape)
    print(df.head())

    ## To regenerate the graph with LLM, set this to True
    regenerate = True

    if regenerate:
        # Todo Stand alle n Paare node_1, node_2 zwischenspeichern
        concepts_list = dataframe_to_graph(df, model='phi3', )  # 'zephyr:latest')
        dfg1 = graph_to_dataframe(concepts_list)

        dfg1.to_csv(out_dir / "graph.csv", sep="|", index=False)
        df.to_csv(out_dir / "chunks.csv", sep="|", index=False)
    else:
        dfg1 = pd.read_csv(out_dir / "graph.csv", sep="|")

    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = 4
    ## Increasing the weight of the relation to 4.
    ## We will assign the weight of 1 when later the contextual proximity will be calculated.
    print(dfg1.shape)
    dfg1.head()

    ## Calculate contextual proximity
    dfg2 = contextual_proximity(dfg1)
    pprint(dfg2.tail())

    ## Merge the two dataframes
    dfg = pd.concat([dfg1, dfg2], ignore_index=True)
    dfg.replace("", np.nan, inplace=True)
    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
        .reset_index()
    )
    pprint(dfg)

    # Calculate the NetworkX Graph
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    print(f'{nodes.shape= }')

    G = nx.Graph()

    ## Add nodes to the graph
    for node in nodes:
        G.add_node(
                str(node)
        )

    ## Add edges to the graph
    for index, row in dfg.iterrows():
        G.add_edge(
                str(row["node_1"]),
                str(row["node_2"]),
                title=row["edge"],
                weight=row['count'] / 4
        )

    # Calculate communities for coloring the nodes
    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    print("Number of Communities = ", len(communities))
    print(communities)

    # Create a dataframe for community color

    palette = "hls"

    ## Now add these colors to communities and make another dataframe
    colors = colors_to_community(communities)
    print(f'{colors= }')

    # Add colors to the graph
    for index, row in colors.iterrows():
        G.nodes[row['node']]['group'] = row['group']
        G.nodes[row['node']]['color'] = row['color']
        G.nodes[row['node']]['size'] = G.degree[row['node']]

    graph_output_directory = "../docs/index.html"

    net = Network(
            notebook=False,
            # bgcolor="#1a1a1a",
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            # font_color="#cccccc",
            filter_menu=False,
    )

    net.from_nx(G)
    # net.repulsion(node_distance=150, spring_length=400)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
    # net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)
    net.show_buttons(filter_=["physics"])

    net.show(graph_output_directory, notebook=False)

