import json

import torch.utils.data as Data


class ReconstrcutDataset(Data.Dataset):
    def __init__(
        self,
        manifest_path,
    ) -> None:
        super().__init__()

        with open(manifest_path) as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        mel
        vgg
        """
        output_data = dict()
