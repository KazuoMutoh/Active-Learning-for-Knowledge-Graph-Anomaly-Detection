import random
import pickle
from graph import KnowledgeGraph


class BaseQueryCreator:

    def __init__(self):
        pass

    def create(self, knowledge_graph:KnowledgeGraph):
        pass

class RandomQueryCreator(BaseQueryCreator):

    def __init__(self):
        super().__init__()

    def create(self, 
               knowledge_graph:KnowledgeGraph, 
               preference='tail',
               k=5):
        
        list_nid = random.sample(list(knowledge_graph.map_id2entity.keys()),k=k)
        list_rid = random.choices(list(knowledge_graph.map_id2relation.keys()),k=k)

        list_query = []
        for nid, rid in zip(list_nid,list_rid):
            
            entity_name = knowledge_graph.map_id2entity[nid]
            entity_text = knowledge_graph.map_entity2text[entity_name]

            relation_name = knowledge_graph.map_id2relation[rid]
            relation_text = knowledge_graph.map_relation2text[relation_name]

            if preference == 'tail':
                entity_role = 'head'
            else:
                entity_role = 'tail'

            dict_query = {
                    f'{entity_role}_name': entity_name,
                    f'{entity_role}_text': entity_text,
                    'relation_name': relation_name,
                    'relation_text': relation_text,
                }
            
            list_query.append(dict_query)

        return list_query


if __name__ == "__main__":

    from graph import DataSet

    #dataset = DataSet.from_pickle('dataset_umls_light.pkl')

    with open('dataset_umls_light.pkl', 'br') as fin:
        dataset = pickle.load(fin)

    query_creator = RandomQueryCreator()
    queries = query_creator.create(dataset.train)

    print(queries)
        