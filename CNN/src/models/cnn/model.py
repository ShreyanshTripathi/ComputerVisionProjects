import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(
        self,
        input_size,
        hidden_layers,
        num_classes,
        activation,
        norm_layer,
        drop_prob=0.0,
    ):
        super(ConvNet, self).__init__()

        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        channels_in = 3
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        for channels_out in self.hidden_layers[:-1]:
            layers.append(
                nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
            )
            layers.append(self.norm_layer(channels_out))
            layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.ReLU())
            if self.drop_prob > 0:
                layers.append(nn.Dropout(self.drop_prob))
            channels_in = channels_out
            print(channels_out)

        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.hidden_layers[-1], self.num_classes))

        self.model = nn.Sequential(*layers)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter.
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img - min) / (max - min)

    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        weights = self.model[0].weight.data.cpu().numpy()

        weights = self._normalize(weights)
        num_filters = weights.shape[0]
        print(num_filters)
        grid_size = int(np.ceil(np.sqrt(num_filters)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

        for i in range(grid_size * grid_size):
            ax = axes[i // grid_size, i % grid_size]
            if i < num_filters:
                filter_img = np.transpose(weights[i], (1, 2, 0))
                ax.imshow(filter_img)
            ax.axis("off")

        plt.show()
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass computations                             #
        # This can be as simple as one line :)
        # Do not apply any softmax on the logits.                                   #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = self.model(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
