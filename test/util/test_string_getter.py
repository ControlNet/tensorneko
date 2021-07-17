import unittest

import torch.nn as nn
from tensorneko.util.string_getter import get_activation, get_loss


class UtilStringGetterTest(unittest.TestCase):

    def test_get_activation(self):
        """Test get_activation function from string to PyTorch module"""
        activation = get_activation("relu")
        self.assertEqual(activation, nn.ReLU)

        activation = get_activation("relu6")
        self.assertEqual(activation, nn.ReLU6)

        activation = get_activation("elu")
        self.assertEqual(activation, nn.ELU)

        activation = get_activation("prelu")
        self.assertEqual(activation, nn.PReLU)

        activation = get_activation("leakyrelu")
        self.assertEqual(activation, nn.LeakyReLU)

        activation = get_activation("selu")
        self.assertEqual(activation, nn.SELU)

        activation = get_activation("tanh")
        self.assertEqual(activation, nn.Tanh)

        activation = get_activation("sigmoid")
        self.assertEqual(activation, nn.Sigmoid)

        activation = get_activation("softplus")
        self.assertEqual(activation, nn.Softplus)

        activation = get_activation("softmax")
        self.assertEqual(activation, nn.Softmax)

        activation = get_activation("logsoftmax")
        self.assertEqual(activation, nn.LogSoftmax)

    def test_get_loss(self):
        """Test get_loss function from string to PyTorch module"""
        loss = get_loss("mseloss")
        self.assertEqual(loss, nn.MSELoss)

        loss = get_loss("nllloss")
        self.assertEqual(loss, nn.NLLLoss)

        loss = get_loss("bceloss")
        self.assertEqual(loss, nn.BCELoss)

        loss = get_loss("bceWithLogitsloss")
        self.assertEqual(loss, nn.BCEWithLogitsLoss)

        loss = get_loss("crossentropyloss")
        self.assertEqual(loss, nn.CrossEntropyLoss)

        loss = get_loss("hingeembeddingloss")
        self.assertEqual(loss, nn.HingeEmbeddingLoss)

        loss = get_loss("kldivloss")
        self.assertEqual(loss, nn.KLDivLoss)

        loss = get_loss("l1loss")
        self.assertEqual(loss, nn.L1Loss)

        loss = get_loss("smoothl1loss")
        self.assertEqual(loss, nn.SmoothL1Loss)

        loss = get_loss("softmarginloss")
        self.assertEqual(loss, nn.SoftMarginLoss)

        loss = get_loss("multilabelsoftmarginloss")
        self.assertEqual(loss, nn.MultiLabelSoftMarginLoss)

        loss = get_loss("multilabelmarginloss")
        self.assertEqual(loss, nn.MultiLabelMarginLoss)
