import os
import sys
import unittest
from sklearn import svm
from schema import SchemaError

# this adds the project root to the PYTHONPATH if its not already there, it makes it easier to run the unit tests
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iris_predict import IrisSVCModel


class TestIrisSVCModel(unittest.TestCase):
    def test1(self):
        """ testing the __init__() method """
        # arrange, act
        model = IrisSVCModel()

        # assert
        self.assertTrue(type(model._svm_model) is svm.SVC)

    def test2(self):
        """ testing the input schema with wrong data """
        # arrange
        data = {'name': 'Sue', 'age': '28', 'gender': 'Squid'}

        # act
        exception_raised = False
        try:
            validated_data = IrisSVCModel.input_schema.validate(data)
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertTrue(exception_raised)

    def test3(self):
        """ testing the input schema with correct data """
        # arrange
        data = {'sepal_length': 1.0,
                'sepal_width': 1.0,
                'petal_length': 1.0,
                'petal_width': 1.0}

        # act
        exception_raised = False
        try:
            validated_data = IrisSVCModel.input_schema.validate(data)
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)

    def test4(self):
        """ testing the output schema with incorrect data """
        # arrange
        data = {'species': 1.0}

        # act
        exception_raised = False
        try:
            validated_data = IrisSVCModel.output_schema.validate(data)
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertTrue(exception_raised)

    def test5(self):
        """ testing the output schema with correct data """
        # arrange
        data = {'species': 'setosa'}

        # act
        exception_raised = False
        try:
            validated_data = IrisSVCModel.output_schema.validate(data)
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)

    def test6(self):
        """ testing the predict() method throws schems exception when given bad data """
        # arrange
        model = IrisSVCModel()

        # act
        exception_raised = False
        try:
            prediction = model.predict({'name': 'Sue', 'age': '28', 'gender': 'Squid'})
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertTrue(exception_raised)

    def test7(self):
        """ testing the predict() method with good data"""
        # arrange
        model = IrisSVCModel()

        # act
        prediction = model.predict(data={'sepal_length': 1.0,
                                         'sepal_width': 1.0,
                                         'petal_length': 1.0,
                                         'petal_width': 1.0})

        exception_raised = False
        try:
            IrisSVCModel.output_schema.validate(prediction)
        except SchemaError as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)
        self.assertTrue(type(prediction) is dict)
        self.assertTrue(prediction["species"] == 'setosa')
        self.assertFalse(exception_raised)


if __name__ == '__main__':
    unittest.main()
