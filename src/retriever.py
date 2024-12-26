import os
from typing import List, Dict, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from settings import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

class LLMRetriever:

    prompt_template = """
    Complete a tail of the triple in [triple].
    The tail should be different from the head.

    [triple]
    {head_name}, {relation_name}, <tail_to_be_completed>

    [Description of head entity]
    {head_text}

    [Description of relaion]
    {relation_text}
    """

    class Tail(BaseModel):
        tail_name: str = \
            Field(
                description="Name of the tail. NONAME if there is no appropriate tail."
            )
        tail_text: str = \
            Field(
                description="Description of the tail. NOTEXT if there is no appropriate tail."
            )
        
    def __init__(self, chat_model_name='gpt-4o'):
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        self.llm = ChatOpenAI(model=chat_model_name, temperature=0.2)
        self.model_with_tools = self.llm.bind_tools([self.Tail])
        self.chain = (self.prompt|self.model_with_tools)

    def complete_triples(self, queries:List[Dict], documents:str=None):
        list_triples = []
        for query in queries:
            response = self.chain.invoke(query)
            print('response')
            print(response.tool_calls[0])
            completed_entity = response.tool_calls[0]["args"]
            triple = query | completed_entity
            list_triples.append(triple)
        return list_triples
    

if __name__ == "__main__":

    import pickle
    from graph import DataSet, KnowledgeGraph
    from query_creator import RandomQueryCreator

    with open('dataset_umls_light.pkl', 'br') as fin:
        dataset = pickle.load(fin)

    query_creator = RandomQueryCreator()
    queries = query_creator.create(dataset.train)

    #print(queries)

    retriever = LLMRetriever()
    triples = retriever.complete_triples(queries)

    #print(triples)
    print('----')
    for triple in triples:
        print(triple)
        print(triple['tail_name'])
        print(triple['tail_text'])
        
