import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device='cuda', num_epochs=1):
    model.to(device)

    train_losses = []
    test_losses = []
    train_acc_total = [] 
    test_acc_total = []
    R2_trainings = []
    R2_tests = []
    best_state_dict = None
    best_loss = float('inf')
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        total_graphs = 0
        K_pred_whole = []
        mu_pred_whole = []
        y_whole_K = []
        y_whole_mu = []
        train_acc_K = [] # accuracy
        train_acc_mu = [] # accuracy
        for data in train_loader:
            K_pred, mu_pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
            data.K = data.K.to(torch.float32)
            data.mu = data.mu.to(torch.float32)
            
            loss_K = criterion(K_pred, data.K.to(device))
            loss_mu = criterion(mu_pred, data.mu.to(device))
            loss = loss_K + loss_mu

            acc_K = 100 - torch.abs((data.K.to(device) - K_pred)/data.K.to(device)) * 100
            acc_mu = 100 - torch.abs((data.mu.to(device) - mu_pred)/data.mu.to(device)) * 100
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # Gradient clipping
            optimizer.step()
            
            num_graphs_in_batch = torch.unique(data.batch).size(0)
            train_loss += loss.item() * num_graphs_in_batch
            total_graphs += num_graphs_in_batch      

            K_pred_whole.append(K_pred.detach().cpu().numpy())
            mu_pred_whole.append(mu_pred.detach().cpu().numpy())
            y_whole_K.append(data.K.cpu().numpy())
            y_whole_mu.append(data.mu.cpu().numpy())
            train_acc_K.append(acc_K.detach().cpu().numpy())
            train_acc_mu.append(acc_mu.detach().cpu().numpy())

        out_whole_K = np.concatenate(K_pred_whole, axis=0)
        out_whole_mu = np.concatenate(mu_pred_whole, axis=0)
        y_whole_K = np.concatenate(y_whole_K, axis=0)  
        y_whole_mu = np.concatenate(y_whole_mu, axis=0)  
        R2_train_K = r2_score(y_whole_K, out_whole_K)
        R2_train_mu = r2_score(y_whole_mu, out_whole_mu)    
        R2_trainings.append((R2_train_K, R2_train_mu))  
        train_acc_K_total = np.concatenate(train_acc_K, axis=0)
        train_acc_mu_total = np.concatenate(train_acc_mu, axis=0)
        train_acc_total.append(np.mean((train_acc_K_total+train_acc_mu_total)/2))
        
        # Append train loss to list
        train_losses.append(train_loss/total_graphs) #len(train_dataset))
        epoch_loss = train_loss/total_graphs#/len(train_loader)

        # Adjust the learning rate based on the loss improvement
        scheduler.step(torch.tensor(epoch_loss).float())

        # Evaluate the model on the test set and save the test loss
        model.eval()

        with torch.no_grad():
            K_pred_whole = []
            mu_pred_whole = []

            y_whole_K = []
            y_whole_mu = []
            test_acc_K = []
            test_acc_mu = []
            test_loss = 0
            total_graphs = 0
            for data in test_loader:
                K_pred, mu_pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
                data.K_Voigt = data.K.to(torch.float32)
                data.mu_Voigt = data.mu.to(torch.float32)
             
                num_graphs_in_batch = torch.unique(data.batch).size(0)
                test_loss += (criterion(K_pred, data.K.to(device).to(torch.float32)).item() + 
                              criterion(mu_pred, data.mu.to(device).to(torch.float32)).item()) * num_graphs_in_batch
                total_graphs += num_graphs_in_batch

                acc_K = 100 - torch.abs((data.K.to(device) - K_pred)/data.K.to(device)) * 100
                acc_mu = 100 - torch.abs((data.mu.to(device) - mu_pred)/data.mu.to(device)) * 100
                K_pred_whole.append(K_pred.cpu().numpy())
                mu_pred_whole.append(mu_pred.cpu().numpy())
                y_whole_K.append(data.K.cpu().numpy())
                y_whole_mu.append(data.mu.cpu().numpy())
                test_acc_K.append(acc_K.detach().cpu().numpy())
                test_acc_mu.append(acc_mu.detach().cpu().numpy())

            K_pred_whole = np.concatenate(K_pred_whole, axis=0)
            mu_pred_whole = np.concatenate(mu_pred_whole, axis=0)
            y_whole_K = np.concatenate(y_whole_K, axis=0)  
            y_whole_mu = np.concatenate(y_whole_mu, axis=0)  
            R2_test_K = r2_score(y_whole_K, K_pred_whole)
            R2_test_mu = r2_score(y_whole_mu, mu_pred_whole)
            R2_tests.append((R2_test_K, R2_test_mu))
            test_loss = test_loss/total_graphs
            test_losses.append(test_loss)  
            test_acc_K_total = np.concatenate(test_acc_K, axis=0)
            test_acc_mu_total = np.concatenate(test_acc_mu, axis=0)
            test_acc_total.append(np.mean((test_acc_K_total+test_acc_mu_total)/2))

        # Check if this model has the best performance so far
        if epoch_loss < best_loss:
            best_loss = test_loss
            best_state_dict = model.state_dict()            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss[Train: {epoch_loss:.3f}, Test: {test_loss:.3f}], R2[train K: {R2_train_K:.3f}, test K: {R2_test_K:.3f}, train mu: {R2_train_mu:.3f}, test mu: {R2_test_mu:.3f}], acc[train K: {np.mean(train_acc_K_total):.2f}, test K: {np.mean(test_acc_K_total):.2f}, train mu: {np.mean(train_acc_mu_total):.2f}, test mu: {np.mean(test_acc_mu_total):.2f}]')

    # Print final test loss
    print(f'Final Test Loss: {test_losses[-1]:.4f}')

    return train_losses, test_losses, R2_trainings, R2_tests, train_acc_total, test_acc_total, best_state_dict
