python train_ci.py --hierarchical_graph ../data/processed_arxiv_dataset/hyper_graph/macro_graph_edges.csv \\
                   --graphs ../data/processed_arxiv_dataset/graphs/ \\
                   --labeled_count 300 \\
                   --first_gcn_dimensions 512 \\
                   --first_dense_neurons 128 \\
                   --second_dense_neurons 2 \\
                   --epochs 400 \\
                   --learning_rate 0.001 \\
                   --budget 100

