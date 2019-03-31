import os
import sys
import unittest

# this adds the project root to the PYTHONPATH if its not already there, it makes it easier to run the unit tests
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model_abc import MLModelException


class TestMLModel(unittest.TestCase):
    def test1(self):
        """ testing the __init__() method """
        # arrange, act
        exception_raised = False
        exception = None
        try:
            raise MLModelException("Testing raising MLModelException.")
        except MLModelException as e:
            exception_raised = True
            exception = e

        # assert
        self.assertTrue(type(exception) is MLModelException)
        self.assertTrue(exception_raised)


if __name__ == '__main__':
    unittest.main()
