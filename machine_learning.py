import numpy as np
import torch
from torch import nn, optim
from torchmetrics.functional import mean_absolute_percentage_error


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input = nn.Linear(2, 6)
        self.hidden_1 = nn.Linear(6, 18)
        self.hidden_2 = nn.Linear(18, 54)

        self.hidden_3 = nn.Linear(54, 108)
        self.hidden_4 = nn.Linear(108, 216)

        self.hidden_5 = nn.Linear(216, 216)

        self.hidden_6 = nn.Linear(216, 108)
        self.hidden_7 = nn.Linear(108, 54)

        self.hidden_8 = nn.Linear(54, 18)
        self.hidden_9 = nn.Linear(18, 6)
        self.output = nn.Linear(6, 2)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.hidden_1(x))
        x = torch.relu(self.hidden_2(x))
        x = torch.relu(self.hidden_3(x))
        x = torch.relu(self.hidden_4(x))
        x = torch.relu(self.hidden_5(x))
        x = torch.relu(self.hidden_6(x))
        x = torch.relu(self.hidden_7(x))
        x = torch.relu(self.hidden_8(x))
        x = torch.relu(self.hidden_9(x))
        return self.output(x)


# def calculate_accuracy(model: nn.Module, x_datapoints, y_datapoints, max_distance: float):
#     with torch.no_grad():
#         n_correct = 0
#         for i, data in enumerate(x_datapoints):
#             inputs = data.reshape(1, -1)
#             labels = y_datapoints[i]
#             outputs = model(inputs)
#             distance = torch.dist(outputs, labels)
#             n_correct += int(distance < max_distance)
#
#         return 100 * n_correct / len(x_datapoints)


def train_model(x_datapoints, y_datapoints, save_as: str, n_epoch: int = 20000, max_distance: float = 10):
    model = NeuralNetwork()
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, verbose=True)

    observations = []
    for epoch in range(n_epoch + 1):
        # forward pass
        scores = model(x_datapoints)
        loss = criterion(scores, y_datapoints)
        # save results
        observations.append([float(epoch), float(loss)])
        # backward pass
        loss.backward()
        # update model
        optimizer.step()
        optimizer.zero_grad()
        # adjust learning rate
        mean_loss = np.mean(np.array(observations, dtype=float)[:, 1])
        scheduler.step(mean_loss)

        if epoch % 1000 == 0:
            # accuracy = calculate_accuracy(model, x_datapoints, y_datapoints, max_distance)
            # error = mean_absolute_percentage_error(scores, y_datapoints)
            error = np.mean(np.clip(np.array([
                float(mean_absolute_percentage_error(scores[index], y_datapoints[index])) * 100
                for index in range(len(scores))
            ]), 0, 100))
            print(f"Epoch {epoch}: Loss = {loss:.6f}, Error = {error:.2f}%")

        # if epoch % 10000 == 0:
        #     points = np.array(observations, dtype=float)
        #     plt.scatter(points[:, 0], points[:, 1], marker=".")
        #     plt.show()

    torch.save(model.state_dict(), save_as)
