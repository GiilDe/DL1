import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple
from .losses import ClassifierLoss
import math


'''

class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO: Create weights tensor of appropriate dimensions
        # Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        normal = torch.distributions.normal.Normal(0, weight_std)
        self.weights = normal.sample(sample_shape=torch.Size([n_features, n_classes]))
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO: Implement linear prediction.
        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.mm(x, self.weights)
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO: calculate accuracy of prediction.
        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = int(((y_pred - y) == 0).sum()) / y_pred.shape[0]
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):

            # TODO: Implement model training loop.
            # At each epoch, evaluate the model on the entire training set
            # (batch by batch) and update the weights.
            # Each epoch, also evaluate on the validation set.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            #
            # Don't forget to add a regularization term to the loss, using the
            # weight_decay parameter.

            total_correct = 0
            average_loss = 0

            # ====== YOUR CODE: ======
            #training set
            n_batches = 0
            for (x, y) in dl_train:
                #actual training
                y_pred, x_scores = self.predict(x)
                ones = torch.ones(1, dtype=x.dtype).repeat(x.shape[0]).reshape(-1, 1)
                x = torch.cat((x, ones), dim=1)
                #statistics
                total_correct += self.evaluate_accuracy(y,y_pred)
                average_loss += loss_fn.loss(x,y,x_scores,y_pred)
                n_batches += 1
                #improving using SGD
                grad = loss_fn.grad()
                grad += (weight_decay*self.weights)
                self.weights -= (learn_rate*grad)
            #add to the epoch statistics
            total_correct = total_correct/n_batches
            average_loss = average_loss/n_batches
            train_res.loss.append(average_loss)
            train_res.accuracy.append(total_correct)

            #validation set
            n_batches = 0
            total_correct = 0
            average_loss = 0
            for (x, y) in dl_valid:
                # prediction
                y_pred, x_scores = self.predict(x)
                # statistics
                total_correct += self.evaluate_accuracy(y, y_pred)
                average_loss += loss_fn.loss(x, y, x_scores, y_pred)
                n_batches += 1
            # add to the epoch statistics
            total_correct = total_correct / n_batches
            average_loss = average_loss / n_batches
            valid_res.loss.append(average_loss)
            valid_res.accuracy.append(total_correct)
            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: Convert the weights matrix into a tensor of images.
        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        weights = self.weights.clone()
        if has_bias:
            weights = weights[:-1]
        w_images = weights.reshape((self.n_classes, *img_shape))
        # TODO: check about the dim
        # ========================

        return w_images


'''


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes
        # TODO: Create weights tensor of appropriate dimensions
        # Initialize it from a normal dist with zero mean and the given std.

        normal = torch.distributions.normal.Normal(0, weight_std)
        self.weights = normal.sample(sample_shape=torch.Size([n_features, n_classes]))
        '''
        sigma = math.sqrt(weight_std)
        rand_weights = torch.randn((n_features, n_classes))
        self.weights = sigma*rand_weights
        '''

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO: Implement linear prediction.
        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.

        y_pred = torch.zeros(len(x), dtype=torch.int64)
        class_scores = x@self.weights

        for i in range(len(x)):
            y_pred[i] = torch.argmax(class_scores[i, :])

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO: calculate accuracy of prediction.
        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.

        equal = len(y) - len(torch.nonzero(y - y_pred))
        acc = equal / len(y)

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):

            # TODO: Implement model training loop.
            # At each epoch, evaluate the model on the entire training set
            # (batch by batch) and update the weights.
            # Each epoch, also evaluate on the validation set.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            #
            # Don't forget to add a regularization term to the loss, using the
            # weight_decay parameter.

            total_correct = 0
            average_loss = 0

            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: Convert the weights matrix into a tensor of images.
        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return w_images


