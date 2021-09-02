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
![avatar]()

## How to obtain arxiv dataset.
