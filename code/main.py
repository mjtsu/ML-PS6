
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import gc
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
import torch.nn.functional as F
import copy
import torchvision.models as models
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# from torchsummary import summary

if __name__ == "__main__":
    df_train = pd.read_csv('../tabular-playground-series-jun-2021/train.csv')
    df_test = pd.read_csv("../tabular-playground-series-jun-2021/test.csv", index_col=0)
    submission = pd.read_csv("../tabular-playground-series-jun-2021/sample_submission.csv")
    submission = submission.set_index('id')

    features = [col for col in df_train.columns if col.startswith('feature_')]
    Y_init = df_train['target'].apply(lambda x: int(x.split("_")[-1]) - 1)
    Y = pd.get_dummies(Y_init)
    scaler = MinMaxScaler()
    all_df = pd.concat([df_train, df_test]).reset_index(drop=True)
    all_df = scaler.fit_transform(all_df[features])
    # X = all_df[:df_train.shape[0]]
    # test_X = all_df[df_train.shape[0]:]
    # X = df_train[features]
    # test_X = df_test

    pca = PCA(n_components=64)
    all_pca = pca.fit_transform(all_df)
    X = all_pca[:df_train.shape[0]].reshape(-1, 1, 8, 8)
    test_X = all_pca[df_train.shape[0]:].reshape(-1, 1, 8, 8)
    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 9)

    if (train_on_gpu):
        model.cuda()
    # summary(model, input_size=(1, 8, 8))

    batch_size = 256
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    train_data = TensorDataset(torch.tensor(np.array(X_train)).to(torch.float64), torch.tensor(np.array(y_train)))
    test_data = TensorDataset(torch.tensor(np.array(X_test)).to(torch.float64), torch.tensor(np.array(y_test)))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=1)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=1)

    lr = 0.00002
    best_model_wts = copy.deepcopy(model.state_dict())
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=5e-4)
    # training params

    # epochs = 50  # 30-40 is approx where I noticed the validation loss stop decreasing
    epochs = 1
    print_every = 1
    clip = 5  # gradient clipping
    best_score = 100
    train_loss = 0
    test_loss = 0
    train_ac = 0
    test_ac = 0
    for e in range(epochs):
        num_correct_train = 0
        for inputs, labels in train_loader:
            model.train()
            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs.float())
            # print(inputs.unsqueeze(0).unsqueeze(0).float().shape)
            # output = F.softmax(output_tmp, dim=1)
            # print(output.squeeze())
            # print(labels.data.max(1, keepdim=True)[1].shape)
            # loss = criterion(output.squeeze(), labels.float())
            loss = criterion(output, labels.data.max(1, keepdim=True)[1].squeeze())
            train_loss = loss
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            pred = output.data.max(1, keepdim=True)[1]
            # print(pred)
            correct_tensor = pred.eq(labels.data.max(1, keepdim=True)[1])
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
                correct_tensor.cpu().numpy())
            num_correct_train += np.sum(correct)
        train_ac = num_correct_train / len(train_data)
        num_correct_test = 0
        val_losses = []
        for inputs, labels in test_loader:
            model.eval()
            num_correct = 0
            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs.float())
            # output = F.softmax(output_tmp, dim=1)
            val_loss = criterion(output, labels.data.max(1, keepdim=True)[1].squeeze())
            # val_loss = criterion(output.squeeze(), labels.float())
            val_losses.append(val_loss.item())
            pred = output.data.max(1, keepdim=True)[1]
            correct_tensor = pred.eq(labels.data.max(1, keepdim=True)[1])
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
                correct_tensor.cpu().numpy())
            num_correct_test += np.sum(correct)
            test_loss = np.mean(val_losses)
        test_ac = num_correct_test / len(test_data)
        if test_loss < best_score:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_score = test_loss
        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Loss: {:.6f}...".format(loss.item()),
              "Test Loss: {:.6f}".format(test_loss),
              "Train Acc: {:.6f}".format(train_ac),
              "Val Acc: {:.6f}".format(test_ac),
              "Best Acc: {:.6f}".format(best_score))

    test_X_tensor = TensorDataset(torch.tensor(np.array(test_X)))
    data_loader = DataLoader(test_X_tensor, shuffle=False, batch_size=batch_size, drop_last=False, num_workers=1)

    results = []
    for inputs in data_loader:
        model.eval()
        if (train_on_gpu):
            inputs = inputs[0].cuda()
        else:
            inputs = inputs[0]
        output_tmp = model(inputs.float())
        output = F.softmax(output_tmp, dim=1)
        for i in range(len(output)):
            results.append(output[i].tolist())
    tmp = pd.DataFrame(results)
    submission = pd.read_csv("../tabular-playground-series-jun-2021/sample_submission.csv")
    submission['Class_1'] = tmp[0]
    submission['Class_2'] = tmp[1]
    submission['Class_3'] = tmp[2]
    submission['Class_4'] = tmp[3]
    submission['Class_5'] = tmp[4]
    submission['Class_6'] = tmp[5]
    submission['Class_7'] = tmp[6]
    submission['Class_8'] = tmp[7]
    submission['Class_9'] = tmp[8]
    submission.to_csv("answer.csv", index=False)
