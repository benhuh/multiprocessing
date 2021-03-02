import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader

from torch.multiprocessing import Pool

class SimpleDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x 
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloader(Din, Dout, total_batch, num_workers, mini_batch):
    ds = SimpleDataSet(x = torch.randn(total_batch, Din),y = torch.randn(total_batch, Dout))
    dl = DataLoader(ds, batch_size=mini_batch, num_workers=num_workers)
    return dl
    
def get_loss(args):
    x, y, model = args
    return mse_loss(model(x), y)


def train():
    num_workers = 2
    Din, Dout = 3, 1
    total_batch = 23
    mini_batch = 2
    num_epochs = 10
    num_procs = 5

    model = nn.Linear(Din, Dout)
    model.share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    dataloader = get_dataloader(Din, Dout, total_batch, num_workers, mini_batch)

    for epoch in range(num_epochs):
        for _, batch in enumerate(dataloader):
            batch_data_model = [(*b, model) for b in batch]
            with Pool(num_procs) as pool:
                optimizer.zero_grad()
                losses = pool.map(get_loss, batch_data_model)

        loss = torch.mean(losses)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    # start = time.time()
    train()
    # print(f'execution time: {time.time() - start}')
