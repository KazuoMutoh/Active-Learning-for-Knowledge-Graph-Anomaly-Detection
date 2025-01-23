from kelpie.link_prediction.models.transe import TransE
import numpy as np
import torch
import logging

prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity. Please determine the correctness of the triple and response True or False.

### Input:
{}

### Response:

"""

# define logger
logger = logging.getLogger(__name__)

class Kopa:

    def __init__(self, tokenizer, adapter, llm_model):
        self.tokenizer = tokenizer
        self.adapter = adapter
        self,llm_model = llm_model

    def _create_prompt(self, h, r, t) -> str:
        raise NotImplementedError
    
    def _embed_prompt(self, prompt: str) -> torch.Tensor:
        raise NotImplementedError
    
    def _create_prefix(self, sample: np.array) -> torch.Tensor:
        raise NotImplementedError
    
    



    


class TransEwithKopa(TransE):

    def __init__(self, *args, tokenizer, adapter, llm_model, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.llm_model = llm_model

    # kopa
    def _create_prompt(self, sample: np.array) -> str:
        return prompt_template.format(f"{self.entity_id_to_name[sample[0]]} \
                                       {self.relation_id_to_name[sample[1]]} \
                                       {self.entity_id_to_name[sample[2]]}")

    def _embed_prompt(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        token_embeds = self.llm_model.model.model.embed_tokens(input_ids)
        return token_embeds
    
    def _create_prefix(self, sample: np.array) -> torch.Tensor:
        struct_h_embed = self.entity_embeddings[sample[0]]
        struct_r_embed = self.entity_embeddings[sample[1]]  
        struct_t_embed = self.entity_embeddings[sample[2]]
        return self.adapter(struct_h_embed, struct_r_embed, struct_t_embed)
    
    def _calculate_score(self, input_embeds) -> float:
        return self.llm_model.generate(
            inputs_embeds=input_embeds, 
            max_new_tokens=16
        )

    #override
    def score(self, samples: np.array) -> np.array:

        scores = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            
            prefix = self._create_prefix(sample)
            
            prompt = self._create_prompt(sample)
            token_embeds = self._embed_prompt(prompt)
            
            input_embeds = torch.cat((prefix, token_embeds), dim=1)

            scores[i] = self._calculate_score(input_embeds)

        return scores
        
        
