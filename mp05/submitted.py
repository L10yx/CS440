import torch
import torch.nn as nn
import torch.nn.functional as func

def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    #raise NotImplementedError("You need to write this part!")
    layers = nn.Sequential(
        nn.Linear(2, 3),
        nn.Sigmoid(),
        nn.Linear(3, 5)
    )
    return layers

def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    #raise NotImplementedError("You need to write this part!")
    loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.MSELoss()
    return loss_function

class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################

        self.hidden = torch.nn.Linear(2883, 320)  # input has 4 values
        self.relu = torch.nn.ReLU()  # activation function
        self.output = torch.nn.Linear(320, 100)
        
        self.dropout = torch.nn.Dropout(0.2)
        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        x_temp = self.hidden(x)             # input data x flows through the hidden layer

        x_temp = self.relu(x_temp)          # use relu as the activation function for intermediate data
        y = self.output(x_temp)        # predicted value
        #y = self.layers(x)
        #raise NotImplementedError("You need to write this part!")
        return y
        ################## Your Code Ends here ##################
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc_input_size = 8 * 1441

        self.fc1 = nn.Linear(self.fc_input_size, 80)
        self.fc2 = nn.Linear(80, 100)

    def forward(self, x):
        x = x.view(-1, 1, 2883)
        x = func.relu(self.conv1(x))
        x = self.pool(x)

        x = x.view(-1, self.fc_input_size)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer

    model = ConvNet()
    loss_fn = create_loss_function()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss}")

    #raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################

    return model
