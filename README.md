# dl4
```python


## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
      running_loss = 0.0
      for images , labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      train_losses.append(running_loss / len(train_loader))
        # Compute validation loss
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for val_images, val_labels in test_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = model(val_images)
            loss = criterion(outputs, val_labels.unsqueeze(1).float())
            val_loss += loss.item()
      val_losses.append(val_loss / len(test_loader))
      model.train()

      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

# Train the model
train_model(model, train_loader,test_loader,num_epochs=10)

# Evaluate the model
test_model(model, test_loader)


```
