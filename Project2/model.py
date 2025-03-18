# model.py

import torch.nn as nn

class TaxiDriverClassifier(nn.Module):
    """
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    """
    def __init__(self, input_dim, output_dim):
        super(TaxiDriverClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_dim, 
            hidden_size=256, 
            num_layers=2,
            dropout=0.3,
            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        lstm_out, _ = self.lstm(x)
        output = lstm_out[:,-1,:]
        x = self.fc(output)

        return x