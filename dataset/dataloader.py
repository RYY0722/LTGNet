from torch.utils.data import DataLoader
from dataset.dataset import TrainSet, ValSet

def get_dataloader(args):
    data_train = TrainSet(args)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    data_test = ValSet(args)
    dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
    dataloader = {'train': dataloader_train, 'val': dataloader_test}

    return dataloader
