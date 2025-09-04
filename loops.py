import torch
from torch import nn
from data import ImageDataset
from torch.utils.data import DataLoader

def training_loop(model, training_data, validation_data, batch = 1, learning_rate = 1e-2, num_epochs = 10, device = "cpu"):

    dataset = ImageDataset(training_data)
    dataloader = DataLoader(dataset, batch_size = batch, shuffle = True)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    model.train()

    for epoch in range(num_epochs):
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

        average_loss = total_loss / len(dataloader)
        print(f'Epoch: {epoch} | Loss: {average_loss}')
        performance_report("Validation", model, validation_data, batch, device)

    return model

def performance_report(type_, model, data, batch_size, device):

    dataset = ImageDataset(data)
    dataloader = DataLoader(dataset, batch_size, shuffle = False)
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    model.eval()

    with torch.no_grad():

        for image, mask in dataloader:
            image = image.to(device)
            mask = mask.to(device)
            logits = model(image)
            loss = loss_function(logits, mask)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(f"{type_} | Loss: {average_loss} \n")

