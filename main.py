from torch.utils.data import DataLoader
from dataloader.DataLoader import LoadItem
from dataloader.Model import SimpleClassification

if __name__ == '__main__':
    data_loader = LoadItem()
    data_interator = DataLoader(data_loader,
                                batch_size=8,
                                shuffle=True)
    # example = next(iter(data_interator))
    # print(example[1][0].shape)
    # model = SimpleClassification()
    # out = model.forward(example[1])
    # print(out)


