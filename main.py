import torch
import torch.nn as nn
import torch.optim as optim

from util import dataload
from models import Covnet, ResNet

from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    save = True ## True : Save Model
    plot = True ## True : plot loss curve & accuracy curve

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : ", device)

    train_set, val_set, test_set = dataload()

    # batch_train
    train_dataset = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    val_dataset = torch.utils.data.DataLoader(dataset=val_set, batch_size=32, shuffle=True)
    test_dataset = torch.utils.data.DataLoader(dataset=test_set, batch_size=32, shuffle=False)

    model_name = "ResNet" ## for savefile
    model = ResNet(nblk_stage1=2, nblk_stage2=2,
                    nblk_stage3=2, nblk_stage4=2)

    model.to(device)

    ## Parameter
    epoch = 300
    learning_rate = 10e-5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    num_train = len(train_set)
    num_val = len(val_set)

    ## Train & Validation
    for i in range(epoch):
        model.train()
        loss_arr = []
        acc_arr = []
        pbar2 = tqdm(train_dataset, unit='batch')
        pbar2.set_description(f'Epoch {i + 1}/{epoch}')
        for ii, (data, label) in enumerate(pbar2):
            images = data.to(device)
            labels = label.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_arr.append(loss.item())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()  ## number of correct answer
            acc_arr.append(correct)

        scheduler.step()

        train_loss.append(np.mean(loss_arr))  ## loss
        train_acc.append(np.sum(acc_arr) / num_train)  ## accuracy
        print('\nEpoch [{}/{}]\nTrain Loss: {:.4f}\tTrain Accuracy: {:.2f}%'.format(i + 1, epoch, np.mean(loss_arr), (
                    np.sum(acc_arr) / num_train) * 100))

        ## validation
        with torch.no_grad():
            model.eval()
            val_loss_arr = []
            val_acc_arr = []
            for j, (data, label) in enumerate(val_dataset):
                val_images = data.to(device)
                val_labels = label.to(device)

                val_output = model(val_images)
                tr_loss = criterion(val_output, val_labels)
                val_loss_arr.append(tr_loss.item())

                # accuracy
                _, predicted = torch.max(val_output.data, 1)
                correct = (predicted == val_labels).sum().item()
                val_acc_arr.append(correct)

            val_loss.append(np.mean(val_loss_arr))
            val_acc.append(np.sum(val_acc_arr) / num_val)

            print('Validation Loss: {:.4f}\tValidation Accuracy: {:.2f}%'.format(np.mean(val_loss_arr), (
                        np.sum(val_acc_arr) / num_val) * 100))

    if save:
        dir = os.path.dirname(__file__)
        model_dir = os.path.join(dir, model_name+'.pt')
        torch.save(model.state_dict(), model_dir)

    if plot:
        dir = os.path.dirname(__file__)
        fold_dir = os.path.join(dir, model_name)
        os.makedirs(fold_dir)
        loss_curve_dir = os.path.join(fold_dir, 'Loss Curve.png')
        accuracy_curve_dir = os.path.join(fold_dir, 'Accuracy Curve.png')

        ## Loss Curve
        plt.plot(val_loss, 'g-')
        plt.plot(train_loss, 'r-')
        plt.legend(['Validation loss', 'Train loss'])
        plt.title('Loss curve')
        plt.grid(True)
        plt.savefig(loss_curve_dir)
        plt.close()
        plt.show()

        ## Accuracy Curve
        plt.plot(val_acc, 'g-')
        plt.plot(train_acc, 'r-')
        plt.legend(['Validation loss', 'Train loss'])
        plt.title('Accuracy curve')
        plt.grid(True)
        plt.savefig(accuracy_curve_dir)
        plt.close()
        plt.show()


if __name__ == '__main__':
    main()