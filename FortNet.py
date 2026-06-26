import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, random_split, DataLoader
import tqdm 

class FortNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(100,100)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(100, 200)
        self.linear_3 = nn.Linear(200, 20)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self,x):
        output =  self.linear_3(self.linear_2(self.activation(self.linear_1(x))))
        print(output.shape)
        
        return output
    
    
    
if __name__ == "__main__":
    model =  FortNet()

    device =  "cuda:0" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    
    data = torch.rand(10,40,100)

    labels = torch.randint(20,(10,40))

    train_size = int(0.8*len(data))
    
    test_size = len(data) - train_size

    dataset =  TensorDataset(data,labels)

    train_dataset, test_dataset =  random_split(dataset,[train_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    #hyperparameters 
    lr = 10e-5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 10 
    

    def trainer(loader, epochs):
       for epoch in tqdm.tqdm(range(epochs)):
        total_loss = 0
        correct = 0
        total = 0 
        
        for X, y in loader:
            X =  X.to(device)
            y = y.to(device)
            print(X.shape, y.shape)
        
        
            optimizer.zero_grad()
            output = model(X)
            output = output.permute(0,2,1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if epoch %2 ==0:
                print(loss.item())
            total_loss += loss.item()
            
            preds = output.argmax(dim=2)
            
            correct = (preds == y).sum().item()
            
            total += y.numel()
            
        accuracy = correct/total
        print(f"Accuracy {accuracy:.4f} | Total loss {total_loss:.4f}")
    trainer(train_loader,epochs)

        
        