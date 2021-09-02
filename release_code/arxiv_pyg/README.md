## How to run
For catious iteration:
```
cd arixv_pyg
./run_ci.sh
```
For active iteration:
```
cd arixv_pyg
./run_ai.sh
```

## What is arxiv dataset.
ArXiv is an open-access repository of electronic preprints. We collect 4666 Computer Science (CS) arXiv papers indexed by Microsoft Academic Graph (MAG). These papers belong to five subject areas including AI, CL, IT, LG and CV. The arXiv papers data forms a citation network which indicates citation relationship. Each paper consists of two parts: title and abstract. The statistics of the arXiv paper dataset is listed in the following table:

|Class|Number|Length of title|Length of abstract|
|-----|------|---------------|------------------|
|AI|232|8.57|158.04|
|CL|648|9.25|140.36|
|IT|909|10.12|170.29|
|LG|1157|8.69|160.63|
|CV|1720|9.41|170.54|


We obtain the hierarchical graph from the arXiv papers data as follows. The skeleton (i.e., edges among graph instances) of the hierarchical graph is provided by citation relations between papers. Each graph instance is a textual graph instance constructed from paper’s textual context including its title and abstract. The following figure illustrates this hierarchical graph. More details can refer to our paper.


![avatar](https://github.com/kochsnow/SEAL_MASTER/blob/main/release_code/arxiv_pyg/text_hg.png?raw=true)

## How to obtain arxiv dataset.
Download [tar file](), unzip it at the directory of "data" and name it as processed_arxiv_dataset.
There are two directories at processed_arxiv_dataset including graphs and hyper_graph.

The hyper_graph has two files indices.mat and macro_graph_edges.csv: indices.mat is used for split train, val and test set;
macro_graph_edges.csv contains citation network for the hierarchical graph.

The directory of graphs includes 4666 json files, and each json file contains the nodes features, edge index list and graph label for each text graph instance.
