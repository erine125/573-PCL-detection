import pytorch
from torch import nn

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out

        num_epochs = 1000  # 1000 epochs
        learning_rate = 0.001  # 0.001 lr

        input_size = 5  # number of features
        hidden_size = 2  # number of features in hidden state
        num_layers = 1  # number of stacked lstm layers

        num_classes = 1  # number of output classes

        lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers,
                      X_train_tensors_final.shape[1])  # our lstm class

        criterion = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            outputs = lstm1.forward(X_train_tensors_final)  # forward pass
            optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = criterion(outputs, y_train_tensors)

            loss.backward()  # calculates the loss of the loss function

            optimizer.step()  # improve from loss, i.e backprop
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))