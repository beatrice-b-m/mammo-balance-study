import os
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class ImpromptuDataset(Dataset):
    def __init__(self, df_dict: dict, img_col: str = '__target_path', transform=None):
        self.transform = transform

        # load the dataframes into one merged one
        merge_df = self._merge_dataframes(df_dict['pos'], df_dict['neg'])
        merge_df.reset_index(drop=True, inplace=True)

        # store the image paths and class labels
        self.path_list = list(merge_df[img_col])
        self.label_list = list(merge_df['__class'])

    def _merge_dataframes(self, pos_df: pd.DataFrame, neg_df: pd.DataFrame):
        # set the class labels
        pos_df.loc[:, '__class'] = 1
        neg_df.loc[:, '__class'] = 0

        # merge the pos and neg dataframes, we don't need to shuffle it here
        # since we can do that with the dataloader
        return pd.concat([pos_df, neg_df], ignore_index=True)

    def _load_img(self, img_path):
        with open(img_path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
        
    def to_loader(self, batch_size: int, num_workers: int = 4, 
                  prefetch_factor: int = 4, shuffle: bool = True):
        return torch.utils.data.DataLoader(
            self, 
            batch_size=batch_size, 
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle=shuffle,
        )

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):        
        image = self._load_img(self.path_list[idx])
        label = self.label_list[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# def load_dataset(data_path: str, transform_list, batch_size: int):
#     """
#     function to load dataset from root image folder
#     """
    
#     dataset = datasets.ImageFolder(
#         data_path, 
#         transform=transform_list
#     )

#     # send dataset to dataloader
#     dataloader = torch.utils.data.DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         pin_memory=True,
#         num_workers=4,
#         prefetch_factor=4,
#         shuffle=True,
#         #sampler=DistributedSampler(dataset),
#     )
#     return dataloader

def load_data_to_loader_dict(base_path: str, mask_factor: str, 
                             set_dir_dict: dict, augment_param_dict: dict, 
                             batch_size: int):
    """
    set_dir_dict should be in the form: {
        "train": TRAIN_DIR, 
        "val": VAL_DIR, 
        "test": TEST_DIR
    }
    """
    
    transform_list = get_augment_list(augment_param_dict)
    
    out_data_loader_dict = {}
    for set_name, set_dir in set_dir_dict.items():
        out_data_loader_dict[set_name] = load_dataset(
            os.path.join(base_path, mask_factor, set_dir), 
            transform_list, 
            batch_size
        )
        
    return out_data_loader_dict

def get_augment_list(param_dict: dict):
    augment_list = list()
    
    augment_list.append(v2.ToImage())
    
    for v in param_dict.values():
        if v.get('enabled', False):
            # extract augment details
            func = v.get('func', None)
            func_params = v.get('params', dict())
            prob = v.get('prob', 1.0)
            
            if prob >= 1.0:
                augment_list.append(
                    func(**func_params)
                )
            else:
                augment_list.append(
                    v2.RandomApply(
                        [func(**func_params)],
                        p=prob
                    )
                )
                
    augment_list += [
        v2.ToDtype(
            dtype=torch.float,
            scale=True
        ),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    # return composed list of augments
    return v2.Compose(augment_list)
