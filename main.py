import pandas as pd
from model import *

n_users = 209850
n_items = 221905

model = MatrixFactorization(n_users, n_items)

model.load_state_dict(torch.load("weight.pth", map_location=torch.device("cpu")))

model.eval()

test_data = pd.read_csv("Dataset/interactions_test.csv")

user_list = test_data.user_id.unique()
item_list = test_data.recipe_id.unique()

user2id = {w: i for i, w in enumerate(user_list)}
item2id = {w: i for i, w in enumerate(item_list)}


class Ratings_Datset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = user2id[self.df['user_id'][idx]]
        user = torch.tensor(user, dtype=torch.long)
        item = item2id[self.df['recipe_id'][idx]]
        item = torch.tensor(item, dtype=torch.long)
        rating = torch.tensor(self.df['rating'][idx], dtype=torch.float)
        return user, item, rating


test_loader = DataLoader(Ratings_Datset(test_data), batch_size=512, shuffle=True, num_workers=2)

users, food, r = next(iter(test_loader))

y = model(users, food) * 5
print("ratings", r[:10].data)
print("predictions:", y.flatten()[:10].data)
