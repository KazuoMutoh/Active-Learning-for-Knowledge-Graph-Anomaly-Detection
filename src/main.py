import os
import pickle
import networkx as nx
from openai import OpenAI
from typing import List, Dict, Tuple
from settings import OPENAI_API_KEY
from tqdm import tqdm
from pykeen.triples.triples_numeric_literals_factory import TriplesNumericLiteralsFactory

# variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
client = OpenAI()

# functions

# classes
class DataSet:
    
    def __init__(self):
        pass

    def load(self, dir_triples:str):
        """
        Initilize data set.

        Args:
            dir_triples(str): 
                A direcotry storing triples. 
                It is assumed follwing files in the directory.
                    - entity2text.txt
                    - relation2text.txt
                    - train.tsv
                    - test.tsv
        Return:
            cls
        """

        dict_entity2text = {}
        with open(f'{dir_triples}/entity2text.txt', 'r') as fin:
            for line in fin:
                line = line.replace('\n', '')
                words = line.split('\t')
                dict_entity2text[words[0]] = words[1]
        
        #print(dict_entity2text)

        dict_relation2text = {}
        with open(f'{dir_triples}/relation2text.txt', 'r') as fin:
            for line in fin:
                line = line.replace('\n', '')
                words = line.split('\t')
                dict_relation2text[words[0]] = words[1]

        list_triples_train = []
        set_entity_train  = set()
        with open(f'{dir_triples}/train.tsv', 'r') as fin:
            for line in fin:
                line = line.replace('\n', '')
                words = line.split('\t')

                if words[0] not in dict_entity2text.keys():
                    print(f'entity not listed in entity2text.txt is found:{words[0]}. skip this triple.')
                    continue
                if words[1] not in dict_relation2text.keys():
                    print(f'entity not listed in relation2text.txt is found:{words[1]}. skip this triple.')
                    continue
                if words[2] not in dict_entity2text.keys():
                    print(f'entity not listed in entity2text.txt is found:{words[2]}. skip this triple.')
                    continue

                triple = (words[0], words[1], words[2])
                list_triples_train.append(triple)
                set_entity_train.add(words[0])
                set_entity_train.add(words[2])

        list_triples_test = []
        set_entity_test  = set()
        with open(f'{dir_triples}/test.tsv', 'r') as fin:
            for line in fin:
                line = line.replace('\n', '')
                words = line.split('\t')
                triple = (words[0], words[1], words[2])
                list_triples_test.append(triple)
                set_entity_test.add(words[0])
                set_entity_test.add(words[2])

        dict_entity2text_train = {k: v for k, v in dict_entity2text.items() \
                                  if k in set_entity_train}
        
        dict_entity2text_test = {k: v for k, v in dict_entity2text.items() \
                                  if k in set_entity_test}
        

        self.train = KnowledgeGraph(list_triples_train, 
                                    dict_entity2text_train,
                                    dict_relation2text)
        
        self.test = KnowledgeGraph(list_triples_test, 
                                    dict_entity2text_test,
                                    dict_relation2text)


class KnowledgeGraph(nx.Graph):
    """
    A class to represent a knowledge graph using NetworkX.
    """

    def __init__(self, 
                 list_triples: List[Tuple],
                 map_entity2text: Dict,
                 map_relation2text: Dict):
        
        super().__init__()

        self.map_entity2text = map_entity2text
        self.map_relation2text = map_relation2text
        
        self.map_id2entity = {}
        self.map_entity2nid = {}
        for nid, (entity, text) in enumerate(map_entity2text.items()):
            self.add_node(nid, name=entity, text=text)
            self.map_id2entity[nid] = entity
            self.map_entity2nid[entity] = nid

        self.map_id2relation = {}
        self.map_relation2id = {}
        for rid, relation in enumerate(map_relation2text.keys()):
            self.map_relation2id[relation] = rid
            self.map_id2relation[rid] = relation

        for h, r, t in list_triples:
            nid_h = self.map_entity2nid[h]
            nid_t = self.map_entity2nid[t]
            self.add_edge(nid_h, nid_t, 
                          name=r, text=map_relation2text[r], 
                          rid=self.map_relation2id[r])

    def set_text_embedding(self, 
                           embedding_model:str = 'text-embedding-3-small', 
                           node_ids: List[int] = None):
        """
        Set text embeddings for specified nodes and their connected edges using a specified embedding model.
        """
        if node_ids is None:
            node_ids = list(self.nodes)

        set_target_nid = set()
        for nid in node_ids:
            if nid in self.nodes:
                set_target_nid.add(nid)
                

        print('calculate text embedding for relations connected to specified entities.')
        set_target_rid = set()
        for nid in node_ids:
            if nid in self.nodes:
                for neighbor in self.neighbors(nid):
                    if self.has_edge(nid, neighbor):
                        rid = self.edges[nid, neighbor]['rid']
                        set_target_rid.add(rid)

        print('calculate text embedding for specified entities.')            
        self.map_nid2textemb = {}
        for nid in tqdm(set_target_nid):
            text = self.map_entity2text[self.map_id2entity[nid]]
            try:
                embedding = self._get_embedding(text, embedding_model)
            except Exception as e:
                print(f'Exception occurs when embedding {nid}:{text}\n{e}')
            self.map_nid2textemb[nid] = embedding

        print('calculate text embedding for relations connected to specified entities.')
        self.map_rid2textemb = {}
        for rid in tqdm(set_target_rid):
            text = self.map_relation2text[self.map_id2relation[rid]]
            try:
                embedding = self._get_embedding(text, embedding_model)
            except Exception as e:
                print(f'Exception occurs when embedding {rid}:{text}\n{e}')
            self.map_rid2textemb[nid] = embedding
       
    def _get_embedding(self, 
                       text: str, 
                       embedding_model:str = 'text-embedding-3-small'):
        """
        Get text embedding using OpenAI's API.
        """
        response = client.embeddings.create(input=[text], model=embedding_model)
        embedding = response.data[0].embedding
        return embedding

    def to_pykeen_triples_factory(self):
        """
        Convert the knowledge graph to a PyKEEN NumericTriplesFactory with embeddings.
        """
        triples = []
        for u, v, data in self.edges(data=True):
            h = u
            t = v
            r = data['name']
            triples.append((h, r, t))

        # Create a NumericTriplesFactory
        triples_factory = TriplesNumericLiteralsFactory.\
            from_labeled_triples(triples, )

        # Add embeddings to the triples factory
        #entity_embeddings = {nid: self.nodes[nid]['embedding'] for nid in self.nodes}
        #relation_embeddings = {data['name']: self.edges[u, v]['embedding'] for u, v, data in self.edges(data=True)}
        #triples_factory.entity_embeddings = entity_embeddings
        #triples_factory.relation_embeddings = relation_embeddings

        return triples_factory

'''
def main():

    # load data set
    # Knowlegde

    # train knoweldge graph embedding
    # model = EmbeddingModel()
    # train(model, triples_factory)

    # create query
    #budget = 10
    #score = 0
    #while budget > 0:
    #    queries = create_query(model)
    #    triples_to_add = retrieve_information(queries, documents)
    #    update_embedding_model(model, triples_to_add)
    #    score = calculate_score(model)
    #    budget -= len(queries)
'''
# %%
if __name__ == "__main__":
    #%%
    dataset = DataSet()
    dataset.load('./data/umls')

    dataset.train.set_text_embedding()

    for nid, node in list(dataset.train.nodes(data=True))[:2]:
        print(nid, node)

    #print(nx.to_pandas_edgelist(dataset.train).head())

    #print(dataset.train.map_nid2textemb)
    #print(dataset.train.map_rid2textemb)

    with open('dataset.pkl', 'bw') as fout:
        pickle.dump(dataset, fout)

    '''
    with open('dataset.pkl', 'br') as fin:
        dataset1 = pickle.load(fin)

    for nid, node in list(dataset1.train.nodes(data=True))[:2]:
        print(nid, node)

    print(nx.to_pandas_edgelist(dataset1.train).head())

    print(dataset1.train.map_nid2textemb)
    print(dataset1.train.map_rid2textemb)
    '''
    

    
