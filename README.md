# Active-Learning-for-Knowledge-Graph-Anomaly-Detection
## Overview
s
## Installation
Clone this repository and move to a top direcotry and execute docker compose.
```bash
docker compose up
```

## Usage
```python
from pykeen.models.multimodal.distmult_literal import DistMultLiteral

# Create knowledge graphs from files.
dataset = DataSet()
dataset.from_files('./data/FB13')

# Set text embeddings for each entities and relations in 
# the knowledge graphs using OpenAI API. (it will takes time) 
dataset.set_text_embeddings()

# Train knowledge graph embedding
dataset.train_graph_embedding(model=DistMultLiteral)
print(f'graph embedding score: {kg_train.get_embedding_score()}')

# Create query
quey_creator = QueryCreator()
queries = quey_creator.create(dataset.train)

# Retrieve triples from documents
retriever = LLMRetriever(dir_docs='./data/documents/FB13')
triples = retriever.retrieve_triple_from_query(queries)

# Update the knowledge graph and its embeddings.
dataset.train.add_triples(triples)
dataset.update_graph_embedding()
print(f'graph embedding score: {kg_train.get_embedding_score()}')
```

## Dependency
+ [pykeen](https://github.com/pykeen/pykeen?tab=readme-ov-file)
+ [Kelpie](https://github.com/AndRossi/Kelpie/tree/master)
+ 

## Data Set
The data set is imported from [KG-BERT](https://github.com/yao8839836/kg-bert/tree/master)
```
./data
```

## Text embedding (Not required)
Although any text embedding can be used, in this implementation, Alpaca is used because KoPA (described below) is selected for knowledge graph embedding which uses Alpaca.

## Knowledge Graph Embedding Model
In this study, [KoPA](https://arxiv.org/pdf/2310.06671) is used for the triple classification. In KoPA, Alpaca is fine-tuned for the triples




## Knowledge Graph Embedding Models
In this study, it is assumed that knowledge each node in knowledge graph has a textual representation. 

In this study, we use [KoPA](https://arxiv.org/pdf/2310.06671) for knowledge graph embedding.


[Large language model enhanced knowledge representation learning survey](https://arxiv.org/pdf/2407.00936)



## Identification of important relations
### Kelpie
[Rossi, Andrea, et al. "Explaining link prediction systems based on knowledge graph embeddings." Proceedings of the 2022 international conference on management of data. 2022.](https://iris.uniroma1.it/bitstream/11573/1640602/4/Rossi_Explaining-link-prediction_2022.pdf)

> ChatGPTのみでもできるのでは？