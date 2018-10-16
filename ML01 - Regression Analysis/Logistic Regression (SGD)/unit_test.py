from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from unittest import TestCase
import unittest
import sgd

class SGDTest(unittest.TestCase):
    """SGD for logistic regression test class.
    """

    def test_logistic(self):
        """Tests the logistic function.
        """
        self.assertAlmostEqual(logistic(1),  0.7310585786300049)
        self.assertAlmostEqual(logistic(2),  0.8807970779778823)
        self.assertAlmostEqual(logistic(-1),  0.2689414213699951)

    def test_dot(self):
        """Tests the dot product.
        """
        d = dot([1.1,2,3.5], [-1,0.1,.08])
        self.assertAlmostEqual(d, -.62)

    def test_predict(self):
        """Tests the predict function.
        """
        model = [1,2,1,0,1]
        point = {'features':[.4,1,3,.01,.1], 'label': 1}
        p = predict(model, point)
        self.assertAlmostEqual(p, 0.995929862284)

    def test_accuracy(self):
        """Tests the accuracy calculation.
        """
        data = extract_features(load_adult_train_data())
        a = accuracy(data, [0]*len(data))
        self.assertAlmostEqual(a, 0.7636129)

    def test_submission(self):
        """Overall test.
        """
        train_data = extract_features(load_adult_train_data())
        valid_data = extract_features(load_adult_valid_data())
        model = submission(train_data)
        predictions = [predict(model, p) for p in train_data]
        print("Training Accuracy: {0}".format(
            accuracy(train_data, predictions)))
        predictions = [predict(model, p) for p in valid_data]
        print("Validation Accuracy: {0}".format(
            accuracy(valid_data, predictions)))


if __name__ == '__main__':
    unittest.main()
