import torch

def train_transformer(model, train_data, optimizer, criterion, num_epochs, device):
    model = model.to(device)
    best_loss = float('inf')
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        # Training loop
        for src, tgt in train_data:
            src = src.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt[:, :-1])  # Get predictions
            output = output.reshape(-1, output.shape[-1])
            target = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            correct_train += (predicted == target).sum().item()
            total_train += target.size(0)
        
        avg_loss = total_loss / len(train_data)
        accuracy = 100 * correct_train / total_train
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
