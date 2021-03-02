"""
Based on: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

Note: as opposed to the torch.multiprocessing package, processes can use
different communication backends and are not restricted to being executed on the same machine.
"""
import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os

num_epochs = 5
batch_size = 8
Din, Dout = 10, 5
data_x = torch.randn(batch_size, Din)
data_y = torch.randn(batch_size, Dout)
data = [(i*data_x, i*data_y) for i in range(num_epochs)]

class OneDeviceModel(nn.Module):
    """
    Toy example for a model ran in parallel but not distributed accross gpus
    (only processes with their own gpu or hardware)
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(Din, Dout)

    def forward(self, x):
        return self.linear(x)

def setup_process(rank, world_size, backend='gloo'):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library/API for process to communicate/coordinate with each other/master. It's a backend library.
    """
    # set up the master's ip address so this child process can coordinate
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
    if torch.cuda.is_available():
        backend = 'nccl'
    # Initializes the default distributed process group, and this will also initialize the distributed package.
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    """ Destroy a given process group, and deinitialize the distributed package """
    dist.destroy_process_group()

def run_parallel_training_loop(rank, world_size):
    """
    Distributed function to be implemented later.

    This is the function that is actually ran in each distributed process.

    Note: as DDP broadcasts model states from rank 0 process to all other processes in the DDP constructor,
    you don’t need to worry about different DDP processes start from different model parameter initial values.
    """
    print()
    print(f"Start running DDP with model parallel example on rank: {rank}.")
    print(f'current process: {mp.current_process()}')
    print(f'pid: {os.getpid()}')
    setup_process(rank, world_size)


    # create model and move it to GPU with id rank
    model = OneDeviceModel().to(rank) if torch.cuda.is_available() else OneDeviceModel().share_memory()
    # ddp_model = DDP(model, device_ids=[rank])
    ddp_model = DDP(model)

    for batch_idx, batch in enumerate(data):
        x, y = batch
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = ddp_model(x)
        labels = y.to(rank) if torch.cuda.is_available() else y
        # Gradient synchronization communications take place during the backward pass and overlap with the backward computation.
        loss_fn(outputs, labels).backward()  # When the backward() returns, param.grad already contains the synchronized gradient tensor.
        optimizer.step()  # TODO how does the optimizer know to do the gradient step only once?

    print()
    print(f"Start running DDP with model parallel example on rank: {rank}.")
    print(f'current process: {mp.current_process()}')
    print(f'pid: {os.getpid()}')
    # Destroy a given process group, and deinitialize the distributed package
    cleanup()

def main():
    print()
    print('running main()')
    print(f'current process: {mp.current_process()}')
    print(f'pid: {os.getpid()}')
    # args
    world_size = mp.cpu_count()
    print('world_size', world_size)
    mp.spawn(run_parallel_training_loop, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    print('starting __main__')
    main()
    print('Done!\a\n')
