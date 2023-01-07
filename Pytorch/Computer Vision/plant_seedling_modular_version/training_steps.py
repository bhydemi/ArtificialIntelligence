import torch 
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics import F1Score
import numpy as np 
from configuration import device
from torch.utils.data import DataLoader


def calculate_class_weights(train_loader: DataLoader):
    """
    This functions takes in a train_loader object and calc
    """
    targets = torch.tensor([])
    for batch, (X, y) in enumerate(train_loader):
        targets = torch.cat((targets, y), 0)

    class_weights=compute_class_weight(class_weight='balanced' , classes = np.unique(targets), y = targets.numpy())
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    return class_weights

def train_model(model: torch.nn.Module, train_loader: DataLoader, loss_criteron, optimizer: torch.optim, classes: list):
    model.train()
    loss_sum = 0
    total_correct = 0 
    f1 = F1Score(task="multiclass", num_classes=len(classes), average='micro' ).to(device)
    pred =  torch.tensor([]).to(device)
    target =  torch.tensor([]).to(device)

    for batch, (X, y) in enumerate(train_loader):
        y_logits =  model(X.to(device))
        loss = loss_criteron(y_logits, y.to(device))
        y_pred = torch.argmax(torch.softmax(y_logits, dim=1), 1)
        pred = torch.cat((pred, y_pred),0)
        target = torch.cat((target, y.to(device)), 0)
        loss_sum += loss.to('cpu').item()
        total_correct += torch.sum(torch.eq(y_pred, y.to(device))).to('cpu').item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    f1_score = f1(pred, target)
    accuracy = total_correct/len(train_loader.dataset)
    avg_loss = loss_sum/len(train_loader.dataset)
    return accuracy, avg_loss, f1_score.item()

def val(model: torch.nn.Module, test_loader: DataLoader, loss_criteron, classes: list):
    model.eval()
    pred =  torch.tensor([]).to(device)
    target =  torch.tensor([]).to(device)
    loss_sum = 0
    total_correct = 0 
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_loader):
            y_logits =  model(X.to(device))
            loss = loss_criteron(y_logits, y.to(device))
            y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            pred = torch.cat((pred, y_pred),0)
            target = torch.cat((target, y.to(device)), 0)
            loss_sum += loss.to('cpu').item()
            total_correct += torch.sum(torch.eq(y_pred, y.to(device))).to('cpu').item()
    f1 = F1Score(task="multiclass", num_classes=len(classes), average='micro' ).to(device)
    f1_score = f1(pred, target)    
    accuracy = total_correct/len(test_loader.dataset)
    avg_loss = loss_sum/len(test_loader.dataset)
    return accuracy, avg_loss, f1_score.item() 
