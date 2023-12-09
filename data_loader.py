from torchvision import datasets, transforms
import torch
import json
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchsampler import ImbalancedDatasetSampler
from torchvision.transforms import autoaugment

def load_data(json_path, batch_size, train, num_workers, domain, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.RandomCrop(224),
#              transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
             transforms.RandomHorizontalFlip(),
#              transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.1),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }
#     data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    data = TBDataset(json_path, tfm = transform['train' if train else 'test'], mode = 'train' if train else 'test', domain = domain)
    data_loader = get_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)

    return data_loader


def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False, **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)


class TBDataset(Dataset):
    def __init__(self, json_path, tfm = None, mode='train', domain='source'):
        cls_dict = {'NM':0, 'TB':1, 'NTN':2}
        self.img_paths = []
        self.labels = []
        self.tfm = tfm
        with open(json_path,'r') as f:
            data = json.load(f)[mode][domain]
        f.close()
        for key in data:
            self.img_paths.extend(data[key])
            self.labels.extend([cls_dict[key]]*len(data[key]))
            
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        if self.tfm is not None:
            image = self.tfm(image)
        return image, self.labels[index] #, self.img_paths[index].split('/')[-1]
    
    def __len__(self):
        return len(self.labels)
    
    def get_labels(self):   
        return self.labels 
    

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)

#         sampler = ImbalancedDatasetSampler(dataset)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0 # Always return 0