## How to run
For activate iteration
```
python train_ai.py --dataset=qq_gc --model=seal --seed=2000 --alpha=0.005 --T=100 --B=500 --epoch=5
```
For cautious iteration
```
python train_ci.py --dataset=qq_gc --model=seal --seed=2000 --alpha=0.005 --T=10 --lam=10 --epoch=5 
```
## what is qq group
For details of qq group, you can see our [previous WWW2019 paper](https://arxiv.org/pdf/1904.05003.pdf) about hierchical grpah in the section 4.3


## How to obtian qq group data
we can obtain the qq group data from [here](https://drive.google.com/drive/folders/1kDF1WxwbWT0bALX1corEW5uql7mUyZJ7?usp=sharing)
There are five files which are all Pickle files:
group_feature_without_zero: list of array, and each array consist of 10-dim user features for one qq group;
group_tuopu_without_zero: list of sparse array, and each sparse array represent edges of one qq group graph instance;
ind.group.graph: it contains the topology information of hierarchical graph, and each node in hierarchical graph is the qq group. 
ind.group.ally: it contains the label information of nodes in the hierarchical graph. And there are two category for qq group including game and non-game.
