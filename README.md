# Active-Learning-for-Knowledge-Graph-Anomaly-Detection

# Installation
Clone this repository and move to a top direcotry and execute docker compose.
```bash
docker compose up
```

# Usage
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

## Data Set
The data set is imported from [KG-BERT](https://github.com/yao8839836/kg-bert/tree/master)
```
./data
```