import torch, random, math, json
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights
import torch.nn as nn
import torch.optim as optim

DTYPE=torch.float32
DEVICE=torch.device("cpu")

    
def chess_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=8*8*15, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=1)
    )
    return model
    
    
###########################################################################################
def trainmodel():

    model = chess_model()
    #model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(in_features=8*8*15, out_features=1))

    #model = torch.nn.Sequential(torch.nn.Flatten(), ChessModel())
    # ... and if you do, this initialization might not be relevant any more ...
    #model[1].weight.data = initialize_weights()
    #model[1].bias.data = torch.zeros(1)

    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
#     for epoch in range(20):
#         for x,y in trainloader:
#             pass # Replace this line with some code that actually does the training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        running_loss = 0.0
        for inputs, labels in trainloader:
           
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()   # Multiply by batch size
        epoch_loss = running_loss / len(trainset)  # Divide by total dataset size
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
        
    # ... after which, you should save it as "model_ckpt.pkl":
    torch.save(model, 'model_ckpt.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()
    
    