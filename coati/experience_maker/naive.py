import torch
from coati.models.utils import compute_reward, normalize

from .base import Experience, ExperienceMaker

import time

from memfrag_tracker import memfrag_tracker_reset, memfrag_tracker_print, get_peak_fragmentation

class MemTrack():
    
    max_memory_reserved = 0
    max_memory_allocated = 0
    max_memory_active = 0
    fragmentation_rate = 0.0
    memory_reserved = 0
    memory_allocated = 0
    memory_reserved_after_empty_cache = 0
    empty_cache_time = 0.0
    
    max_memory_reserved_0 = 0
    max_memory_allocated_0 = 0
    max_nrm = 0

    @staticmethod
    def reset_peak_memory_stats():
        torch.cuda.reset_peak_memory_stats()
        empty_cache_time = 0
        memfrag_tracker_reset()

    @staticmethod
    def _get_memory_stats():
        MemTrack.max_memory_reserved = torch.cuda.max_memory_reserved()
        MemTrack.max_memory_allocated = torch.cuda.max_memory_allocated()
        MemTrack.max_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0) 
        if MemTrack.max_memory_reserved > 0:
            MemTrack.fragmentation_rate = (MemTrack.max_memory_reserved - MemTrack.max_memory_allocated) / MemTrack.max_memory_reserved
        else:
            MemTrack.fragmentation_rate = 0
        MemTrack.memory_reserved = torch.cuda.memory_reserved()
        MemTrack.memory_allocated = torch.cuda.memory_allocated()   
       
    @staticmethod 
    def _get_memory_stats_0():
        MemTrack.max_memory_reserved_0 = max(MemTrack.max_memory_reserved_0, MemTrack.max_memory_reserved)
        MemTrack.max_memory_allocated_0 = max(MemTrack.max_memory_allocated_0, MemTrack.max_memory_allocated)
        MemTrack.max_nrm = max(MemTrack.max_nrm, MemTrack.max_memory_reserved - get_peak_fragmentation())
    
    @staticmethod
    def _empty_cache(title: str):
        # if title is "Critic Optimizer": # step
        # if title is "Generation" or "Inference" in title: # inference
        # if "Forward" in title or "Backward" in title or "Optimizer" in title: # training
        if False:
        # if True:
            start_time = time.time()
            torch.cuda.empty_cache()
            end_time = time.time()
            MemTrack.empty_cache_time = (end_time - start_time) * 1000.0
        else:
            MemTrack.empty_cache_time = 0.0
    
    @staticmethod
    def _get_memory_info_after_empty_cache():
        MemTrack.memory_reserved_after_empty_cache = torch.cuda.memory_reserved()
        
    @staticmethod
    def collect_and_print_memory_stats(title: str):
        MemTrack._get_memory_stats()
        if title != "Generation":
            MemTrack._get_memory_stats_0()
        # print(f"new rm = {MemTrack.max_memory_reserved - get_peak_fragmentation()}", flush=True)
        MemTrack._empty_cache(title)
        MemTrack._get_memory_info_after_empty_cache()
        # print(f"{title},{MemTrack.max_memory_reserved},{MemTrack.max_memory_allocated},{MemTrack.max_memory_active},{MemTrack.fragmentation_rate * 100:.2f}%,\
        #     {MemTrack.memory_reserved},{MemTrack.memory_allocated},\
        #         {MemTrack.memory_reserved_after_empty_cache},{MemTrack.empty_cache_time}", flush=True)
        # memfrag_tracker_print()
        if title != "Generation":
            print(f"{title},{MemTrack.max_memory_reserved_0/1073741824:.1f},{MemTrack.max_memory_allocated_0/1073741824:.1f}, ,{MemTrack.max_nrm/1073741824:.1f}", flush=True)


sequences, attention_mask, action_mask = None, None, None

class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    @torch.no_grad()
    def make_experience(self, timestep, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        global sequences, attention_mask, action_mask
        
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        MemTrack.reset_peak_memory_stats()

        # if timestep == 1:
        if True:
            sequences, attention_mask, action_mask = self.actor.generate(input_ids,
                                                                     return_action_mask=True,
                                                                     **generate_kwargs)
        MemTrack.collect_and_print_memory_stats("Generation")
        num_actions = action_mask.size(1)

        MemTrack.reset_peak_memory_stats()
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        MemTrack.collect_and_print_memory_stats("Actor Inference")
        
        MemTrack.reset_peak_memory_stats()
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        MemTrack.collect_and_print_memory_stats("Ref Inference")

        MemTrack.reset_peak_memory_stats()
        value = self.critic(sequences, action_mask, attention_mask)
        MemTrack.collect_and_print_memory_stats("Critic Inference")

        MemTrack.reset_peak_memory_stats()
        r = self.reward_model(sequences, attention_mask)
        MemTrack.collect_and_print_memory_stats("Reward Inference")
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)

        advantage = reward - value
        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)
            
        batch_size = sequences.size(0)
        actions = torch.randint(0, num_actions, (batch_size,)).to(sequences.device)

        return Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask, actions)
