import os
import pickle
import networkx as nx
import numpy as np
from openai import OpenAI
from typing import List, Dict, Tuple
from settings import OPENAI_API_KEY
from tqdm import tqdm
from pykeen.triples.triples_numeric_literals_factory import TriplesNumericLiteralsFactory
from pykeen.models.multimodal.distmult_literal import DistMultLiteral
from pykeen.models.multimodal.base import LiteralModel
from pykeen.pipeline import pipeline

# variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
client = OpenAI()

# functions

# classes
class DataSet:
    
    def __init__(self):
        pass

    def to_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def from_files(self, dir_triples:str):
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
        with open(f'{dir_triples}/entity2textlong.txt', 'r') as fin:
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
        
    def set_text_embeddings(self):

        self.train._set_text_embeddings()
        self.test._set_text_embeddings()

    def train_graph_embeddings(self, model:LiteralModel, **kwargs):

        self.model = model

        triples_train = self.train._to_pykeen_triples_factory()
        triples_test = self.test._to_pykeen_triples_factory()

        self.graph_embedding = \
            pipeline(training=triples_train, testing=triples_test, model=model,**kwargs)
        
    def get_embedding_socre(self):
        return self.graph_embedding.get_metric()
    
    def update_graph_embedding(self, **kwargs):
        self.train_graph_embeddings(self.model, **kwargs)
    

class KnowledgeGraph(nx.Graph):
    """
    A class to represent a knowledge graph using NetworkX.
    """

    def __init__(self, 
                 list_triples: List[Tuple],
                 map_entity2text: Dict,
                 map_relation2text: Dict):
        
        super().__init__()

        self.list_triples = list_triples
        self.map_entity2text = map_entity2text
        self.map_relation2text = map_relation2text

        self._create()

    def _create(self):

        self.num_triples = len(self.list_triples)
        self.num_entities = len(self.map_entity2text)
        self.num_relations = len(self.map_relation2text)
        
        self.map_id2entity = {}
        self.map_entity2nid = {}
        for nid, (entity, text) in enumerate(self.map_entity2text.items()):
            self.add_node(nid, name=entity, text=text)
            self.map_id2entity[nid] = entity
            self.map_entity2nid[entity] = nid

        self.map_id2relation = {}
        self.map_relation2id = {}
        for rid, relation in enumerate(self.map_relation2text.keys()):
            self.map_relation2id[relation] = rid
            self.map_id2relation[rid] = relation

        for h, r, t in self.list_triples:
            nid_h = self.map_entity2nid[h]
            nid_t = self.map_entity2nid[t]
            self.add_edge(nid_h, nid_t, 
                          name=r, text=self.map_relation2text[r], 
                          rid=self.map_relation2id[r])

    def _set_text_embeddings(self, 
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

        self.dim_embedding = len(embedding)

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

    def _to_pykeen_triples_factory(self):
        """
        Convert the knowledge graph to a PyKEEN NumericTriplesFactory with text-embeddings.
        """

        numeric_literals = np.zeros((self.num_entities,self.dim_embedding))
        for nid, embeddings in self.map_nid2textemb.items():
            numeric_literals[nid,:] = embeddings

        print(numeric_literals[0])

        # Create a NumericTriplesFactory
        triples_factory = TriplesNumericLiteralsFactory.\
            from_labeled_triples(self.list_triples, 
                                 numeric_literals)
        
        return triples_factory

    def add_triples(self, triples:List[Dict]):

        list_triples = []
        map_entity2text = []
        for triple in triples:
            
            list_triples.append(\
                (triple['head_name'], triple['relation_name'], triple['tail_name'])
            )

            map_entity2text[triple['tail_name']] = triple['tail_text']

        self.list_triples += list_triples
        self.map_entity2text |= map_entity2text

        self._create()

    def search_entity(self, name:str):
        for nid, node in self.nodes(data=True):
            _name = node.get('name')
            if name is not None and name == _name:
                return node

# %%
if __name__ == "__main__":
    #%%
    init = True

    dataset = DataSet()
    dataset.from_files('./data/umls')
    dataset.to_pickle('dataset_umls_light.pkl')

    '''
    if init:

        dataset.from_files('./data/umls')
        dataset.set_text_embeddings()

        for nid, node in list(dataset.train.nodes(data=True))[:2]:
            print(nid, node)

        dataset.to_pickle('dataset_umls.pkl')
    
    else:

        dataset.from_pickle('dataset_umls.pkl')


    dataset.train_graph_embeddings(DistMultLiteral, training_kwargs={'num_epochs':1})
    '''

        


    
