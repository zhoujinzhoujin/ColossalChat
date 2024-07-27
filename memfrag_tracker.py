import torch
import mlinsight

fragmentation = 0
max_reserved_memory = 0
fragmentation_at_mempeak = 0

def cuMemAllocCB(ptr, size):
    global fragmentation, fragmentation_obj_sizes, max_reserved_memory, fragmentation_at_mempeak
    reserved_memory = torch.cuda.memory_reserved()
    allocated_memory = torch.cuda.memory_allocated()
    
    fragmentation = min(fragmentation, reserved_memory - allocated_memory)
    
    if size <= reserved_memory - allocated_memory:
        fragmentation = reserved_memory - allocated_memory
        
    if reserved_memory >= max_reserved_memory:
        max_reserved_memory = reserved_memory
        fragmentation_at_mempeak = min(fragmentation, fragmentation_at_mempeak)
        
    

def cuMemFreeCB(ptr):
    global fragmentation, fragmentation_obj_sizes, max_reserved_memory, fragmentation_at_mempeak
    reserved_memory = torch.cuda.memory_reserved()
    allocated_memory = torch.cuda.memory_allocated()

    fragmentation = min(fragmentation, reserved_memory - allocated_memory)
    
def memfrag_tracker_init():
    mlinsight.hello()
    mlinsight.install(cuMemAllocCB, cuMemFreeCB)
    
def memfrag_tracker_reset():
    global fragmentation, max_reserved_memory, fragmentation_at_mempeak
    max_reserved_memory = torch.cuda.memory_reserved()
    fragmentation_at_mempeak = fragmentation
    
def memfrag_tracker_print():
    global max_reserved_memory, fragmentation_at_mempeak
    print(f"Reserved memory={max_reserved_memory}, fragmentation={fragmentation_at_mempeak}", flush=True)
    
def get_peak_fragmentation():
    return fragmentation_at_mempeak