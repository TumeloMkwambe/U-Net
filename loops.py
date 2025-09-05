import os
import copy
import torch
import wandb
import numpy as np
from torch import nn
from data import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def metrics(masks, predictions):
    '''
    Only computes metrics for foreground pixels (positives).
    '''

    predictions = np.concatenate(predictions).flatten()
    masks = np.concatenate(masks).flatten()
    class_ = 1

    accuracy = accuracy_score(masks, predictions)
    precision = precision_score(masks, predictions, average = 'binary', pos_label = class_, zero_division = 0)
    recall = recall_score(masks, predictions, average = 'binary', pos_label = class_, zero_division = 0)
    f1 = f1_score(masks, predictions, average = 'binary', pos_label = class_, zero_division = 0)

    actual_positives = (masks == class_)
    predicted_positives = (predictions == class_)
    intersection = np.logical_and(actual_positives, predicted_positives).sum()
    union = np.logical_or(actual_positives, predicted_positives).sum()
    iou = intersection / union if union > 0 else 0.0

    return accuracy, precision, recall, f1, iou

def performance_report(model, data, batch_size, device):

    dataset = ImageDataset(data)
    dataloader = DataLoader(dataset, batch_size, shuffle = False)
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    predictions = []
    masks = []
    model.eval()

    with torch.no_grad():

        for image, mask in dataloader:
            image = image.to(device)
            mask = mask.to(device)
            logits = model(image)
            loss = loss_function(logits, mask)

            total_loss += loss.item()
            prediction = torch.argmax(logits, dim = 1)
            predictions.append(prediction.cpu().numpy())
            masks.append(mask.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    accuracy, precision, recall, f1, iou = metrics(masks, predictions)

    return average_loss, accuracy, precision, recall, f1, iou

def training_loop(model, training_data, validation_data, run_name, batch = 1, learning_rate = 1e-2, num_epochs = 10, device = "cpu"):

    dataset = ImageDataset(training_data)
    dataloader = DataLoader(dataset, batch_size = batch, shuffle = True)

    model = model.to(device)
    best_iou = 0.0
    best_model_state = None

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    wandb.init(entity = "computer-vision-wits", project = "U-Net", name = run_name)

    config = wandb.config
    config.epochs = num_epochs
    config.batch_size = batch
    config.learning_rate = learning_rate

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for image, mask in dataloader:
            image = image.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            logits = model(image)
            loss = loss_function(logits, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(dataloader)
        val_loss, accuracy, precision, recall, f1, iou = performance_report(model, validation_data, batch, device)

        wandb.log({
            'Training Loss': train_loss,
            'Validation Loss': val_loss,
            'Validation Accuracy': accuracy,
            'Validation Precision': precision,
            'Validation Recall': recall,
            'Validation F1': f1,
            'Validation IoU': iou
        })

        if best_iou < iou:
            best_iou = iou
            best_model_state = copy.deepcopy(model.state_dict())

        '''
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
        }
        checkpoint_name = os.path.join("checkpoints", f"{run_name}_epoch{epoch}.pth")
        torch.save(checkpoint, checkpoint_name)
        '''

        print(f'Epoch: {epoch}')
        print(f'Training | Loss: {train_loss}')
        print(f"Validation | Loss: {val_loss} | IoU: {iou} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1} \n")        

    wandb.finish()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
