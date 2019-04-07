import abc
import torch
'''

class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        real_scores = x_scores[torch.arange(y.shape[0]), y]
        real_scores = real_scores.repeat(x_scores.shape[1], 1).t()
        loss_matrix = (x_scores - real_scores) + self.delta
        loss_matrix[loss_matrix < 0] = 0
        loss_matrix[torch.arange(y.shape[0]), y] = 0
        loss = torch.mean(loss_matrix.sum(dim=1))
        # TODO: check about the lamda and the Normal
        # ========================


        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['x'] = x
        self.grad_ctx['loss_matrix'] = loss_matrix
        self.grad_ctx['y'] = y
        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        G = self.grad_ctx['loss_matrix'].clone()
        x_t = self.grad_ctx['x'].t()
        y = self.grad_ctx['y']
        G[G > 0] = 1
        G[torch.arange(y.shape[0]), y] = -1 * G.sum(dim=1)
        grad = torch.mm(x_t, G)
        grad = grad/y.shape[0]
        # ========================

        return grad


'''


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).
        loss = None
        
        N = len(x)
        D = len(x[0])
        C = len(x_scores[0])

        ones_vector = torch.ones(size=(1, C), dtype=y.dtype)
        y = y.reshape(N, 1)
        y_m = y@ones_vector
        s_yi = torch.gather(x_scores, dim=1, index=y_m)
        m = (x_scores - s_yi) + self.delta
        m[m < 0] = 0
        L_i_W = m.sum() - self.delta*N
        L_W = (1/N)*L_i_W
        loss = L_W


        # TODO: Save what you need for gradient calculation in self.grad_ctx
        self.grad_ctx['x']=x
        self.grad_ctx['y']=y
        self.grad_ctx['m']=m
        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        N = len(self.grad_ctx['x'])
        y = self.grad_ctx['y']
        x = self.grad_ctx['x']
        G = self.grad_ctx['m']
        G[self.grad_ctx['m'] > 0] = 1
        G[torch.arange(y.shape[0]), y] = -1*(G.sum(dim=1)-1)
        # todo : zero yi cells and delete the +1
        grad = (x.t())@G
        grad = (1/N)*grad

        return grad



