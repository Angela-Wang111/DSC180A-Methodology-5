## Helper class for DataLoader
class ImportData(Dataset):
    def __init__(self, df, max_size):
        
        self.all_path = df.iloc[:,-1].values
        y = df.iloc[:,0].values

        self.y_train=torch.tensor(y,dtype=torch.float32).unsqueeze(1)
        
        self.max_size = max_size
        
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self,idx):        
        
        file_path = self.all_path[idx]
        img = torch.load(file_path)
        img_min = torch.min(img)
        img_max = torch.max(img)
        img_norm = (img - img_min) / (img_max - img_min)
        img = img.expand(3, self.max_size, self.max_size)
              
        return img, self.y_train[idx]
    
    
    
def create_loader(train_path, val_path, test_path, res, batch_size):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    train_ds = ImportData(train_df, res)
    val_ds = ImportData(val_df, res)
    test_ds = ImportData(test_df, res)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
    
    return train_loader, val_loader, test_loader

    
BATCH_SIZE = 8
RES = 224
