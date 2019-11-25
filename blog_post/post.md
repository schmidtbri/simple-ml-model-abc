Title: A Simple ML Model Base Class
Date: 2019-04-02 09:20
Category: Blog
Slug: a-simple-ml-model-base-class
Authors: Brian Schmidt
Summary: When creating software it is often useful to write abstract classes to help define different interfaces that classes can implement and inherit from. By creating a base class, a standard can be defined that simplifies the design of the whole system and clarifies every decision moving forward.

When creating software it is often useful to write abstract classes to 
help define different interfaces that classes can implement and inherit
from. By creating a base class, a standard can be defined that
simplifies the design of the whole system and clarifies every decision
moving forward.

The integration of ML models with other software components is often
complicated and can benefit greatly from using an Object Oriented
approach. Recently, I've been seeing this problem solved in many
different ways, so I decided to try to implement my own solution.

In this post I will describe a simple implementation of a base class for
Machine Learning Models. This post will focus on making predictions with
ML models, and integrating ML models with other software components.
Training code will not be shown to keep the code simple. The code in
this post will be written in Python, if you aren't familiar with
abstract base classes in Python,
[here](https://www.python-course.eu/python3_abstract_classes.php)
is a good place to learn.

## Scikit-learn's Approach to Base Classes

The most well known ML software package in python is scikit-learn, and
it provides a set of abstract base classes in the [base.py
module](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py).
The scikit-learn API is a great place to learn about machine learning
software engineering in general, but in this case we want to focus on
it's approach to base classes for making predictions with ML models.

Scikit-learn defines an abstract base class called Estimator which is
meant to be the base class for any class that is able to learn from a
data set, a class that derives from Estimator must implement a "fit"
method. Scikit-learn also defines a Predictor base class that is meant
to be the base class for any class that is able to infer from learned
parameters when presented with new data, a class that derives from
Predictor must implement a "predict" method. These two bases classes are
some of the most commonly used abstractions in the Scikit-learn package.
By defining these base classes, the Scikit-learn project provides a
strong base for coding ML algorithms.

These two interfaces are broad enough to take us far, but what about
serialization and deserialization? ML models need to be loaded from
storage before they can be used. On this front scikit-learn is mostly
silent, and no standard interface for hiding the details of model
serialization and deserialization is provided. Also, what if we need to
publish schema information about the input and output data that a model
needs for scoring? Scikit-learn does not provide a way to do this
either, since it uses numpy arrays for input and output.

Because of these factors, using Scikit-learn's API is not necessarily
the best way to integrate ML models with other software components.
Integrating a Scikit-learn model with other software components by using
the Scikit-learn API exposes internal details about how the model is
serialized and how information is passed into the model. For example, if
a Data Scientist hands over a scikit-learn model in a pickled file along
with some code, a software engineer would have to be familiar with how
to deserialize the model object and how to structure a Numpy array in
such a way that it will be accepted by the model's predict() method. The
best way to solve this problem is to hide these implementation details
behind an interface.

In summary, to simplify the use of ML models within production systems,
it would be useful to solve a couple of issues:

-   How to consistently and transparently send data to the model

-   How to load serialized model assets when instantiating a model

-   How to publish input and output data schema information

## Some Solutions

Over the last few years, a few big tech companies have been developing
proprietary in-house machine learning infrastructure and software. Some
of these companies sell access to their ML platform and others have
published details about their approach to ML infrastructure. Also, there
have been a few open source projects that seek to simplify the
deployment of ML models to production systems. In this section I will
describe some solutions that have emerged recently for the problems
described above.

### AWS Sagemaker

AWS Sagemaker is a platform for training and deploying ML models within
the AWS ecosystem. The platform has several ready-made ML algorithms
that can be leveraged without writing a lot of code. However, a way to
deploy custom ML code to the platform is provided. To deploy a
prediction endpoint on top of the Sagemaker service, a Python Flask
application with a "/ping" and "/invocations" endpoints must be created
and deployed within a Docker container.

In the Sagemaker example published
[here](https://github.com/awslabs/amazon-sagemaker-examples/blob/35941a33425b3a441275abc7243eb1f959a584e4/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py#L24-L43),
we can see the recommended way to run the model prediction code within
the Flask application. In the example, the scikit-learn model object is
deserialized and saved as a class property, and the model is then
accessed by the "predict" method. This implementation does not provide a
way to publish schema metadata about the model and does not enforce any
specific implementation on the model code. The AWS Sagemaker library
does not provide a base class to help write the model code.

### Facebook

Facebook published a blog post about their ML systems
[here](https://code.fb.com/ml-applications/introducing-fblearner-flow-facebook-s-ai-backbone/).
The FBLearner Flow system is made up of workflows and operators. A
workflow is a single unit of work with a specific set of inputs and
outputs, a workflow is made up of operators which do simple operations
on data. The blog post shows how to train a Decision Tree model on the
iris data set. The blog post does not provide many implementation
details about their internal Python packages. An interesting part of the
approach taken is the fact that schema metadata is attached to every
workflow created, ensuring type safety at runtime. There are not details
about loading and storing model assets. Facebook's FBFlow Python package
does not use base classes that developers can inherit from to write
code, but uses function annotations to attach metadata to ML model code.

### Uber

Uber published a blog post about their approach to custom ML models
[here](https://eng.uber.com/michelangelo-pyml/). Uber's
PyML package is used to deploy ML models that are not natively supported
by Uber's Michelangelo ML platform, which is described
[here](https://eng.uber.com/michelangelo/). The PyML
package does not specify how to write model training code, but does
provide a base class for writing ML model prediction code. The base
class is called DataFrameModel. The interface is very simple, it only
has two methods: the \_\_init\_\_() method, and the predict() method.
The model assets are required to be deserialized in the class
constructor and all prediction code is in the predict method of the
class.

The DataFrameModel interface requires the use of Pandas dataframes or
tensors when giving data to the model for prediction. This is a design
decision can backfire because there is no way to tell the user of the
model how to structure the input data to the model. However, the use of
the \_\_init\_\_() method for loading model assets helps to hide the
complexity of the model from the user. Also, by using base classes that
must be inherited from in order to deploy code to the production
systems, certain requirements can be more easily checked.

### Seldon Core

Seldon Core is an open source project for hosting ML models. It supports
custom Python models, as described
[here](https://docs.seldon.io/projects/seldon-core/en/latest/python/python_component.html).
The model code is required to be in a Python class with an
\_\_init\_\_() method and a predict() method, it follows Uber's design
closely but does not use an abstract base class to enforce the
interface. Another difference is that Seldon allows the model class to
return results in several different ways, and not just in Pandas
dataframes. Seldon also allows the model class to return column name
metadata for the model inputs, but no type metadata.

A Simple ML Model Base Class
============================

NOTE: All of the code shown in this section can be found in [this
Github
repository](https://github.com/schmidtbri/simple-ml-model-abc).

In this section I will present a simple abstract base class that
combines the strengths of the approaches shown above into one abstract
base class for ML models. I will also explain the reasoning behind the
design.

Here is the code for the abstract base class:

```python
class MLModel(ABC):
    """ An abstract base class for ML model prediction code """
    @property
    @abstractmethod
    def input_schema(self):
    raise NotImplementedError()
    
    @property
    @abstractmethod
    def output_schema(self):
    raise NotImplementedError()
    
    @abstractmethod
    def __init__(self):
    raise NotImplementedError()
    
    @abstractmethod
    def predict(self, data):
    self.input_schema.validate(data)

```

The code looks very similar to Uber's and Seldon Core's approach. The
model file deserialization code is still expected to be implemented in
the \_\_init\_\_() method, and the prediction code is still expected to
be in the predict() method. Any model that needs to be used by other
software packages is expected to derive from the MLModel abstract base
class and implement these two methods.

However, there are some differences. The input to the predict method is
not expected to be of any particular type, it can be any Python type as
long as the input data is packaged into a single input parameter called
"data". This is different from Seldon Core's and Uber's approach which
required Numpy arrays and Pandas arrays.

Another difference is that the base class shown above requires the model
creator to attach schema metadata to their implementation. The base
class has two extra properties that are not present in the Seldon Core
and Uber implementations: the "input\_schema" and "output\_schema"
properties are meant to publish the schema of the data that the model
will accept in the predict method and the shema of the model that the
model will output from the predict method. To do this, I will use the
python schema package, but there are many options for writing and
enforcing schema, for example the marshmallow-schema and schematics
python packages.

We also need to define a way for a model creator to raise exceptions.
For this we can write a simple custom Exception:

```python
class MLModelException(Exception):
    """ Exception type for use within MLModel derived classes """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
```

Using the Base Class
====================

This blog post deals purely with the ML code that will be used for
predicting in production and not with the model training code. However,
we still need to have a model to work with. Here's a simple scikit-learn
model training script:

```python
iris = datasets.load_iris()
svm_model = svm.SVC(gamma=0.001, C=100.0)
svm_model.fit(iris.data\[:-1\], iris.target\[:-1\])

dir_path = os.path.dirname(os.path.realpath(__file__))
file = open(os.path.join(dirpath, "model_files", "svc_model.pickle"), 'wb')
pickle.dump(svm_model, file)
file.close()
```

Now that we have a trained model, we can write the class that will inherit from MLModel and make predictions:

```python
class IrisSVCModel(MLModel):
    """ A demonstration of how to use """
    input_schema = Schema({'sepal_length': float,
    'sepal_width': float,
    'petal_length': float,
    'petal_width': float})
    
    # the output of the model will be one of three strings
    output_schema = Schema({'species': Or("setosa",
        "versicolor",
        "virginica")})

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = open(os.path.join(dir_path, "model_files", "svc_model.pickle"), 'rb')
        self._svm_model = pickle.load(file)
        file.close()
    
    def predict(self, data):
        # calling the super method to validate against the
        # input_schema
        super().predict(data=data)
        
        # converting the incoming dictionary into a numpy array
        # that can be accepted by the scikit-learn model
        X = array([data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]]).reshape(1, -1)
        
        # making the prediction
        y_hat = int(self._svm_model.predict(X)[0])
        
        # converting the prediction into a string that will match
        # the output schema of the model, this list will map the
        # output of the scikit-learn model to the string expected by
        # the output schema
        targets = ['setosa', 'versicolor', 'virginica']
        species = targets[y_hat]
        
        return {"species": species}
```

One useful thing about using the schema package for building the input
and output schemas of the model is that it supports exporting the schema
in the JSON schema format:

```python
>>> model = IrisSVCModel()
>>> print(json.dumps(model.input_schema.json_schema("https://example.com/my-schema.json")))
{"type": "object", "properties": {"sepal_length": {"type": "number"}, "sepal_width": {"type": "number"},
...
...
```

## Conclusion

In this post I showed a few different approaches to deploying ML model
code to production systems. I also showed an implementation of a Python
base class that brings together the best features of the different
approaches discussed. In conclusion I will discuss some of the benefits
of the approach I sketched out above.

The MLModel base class has very few dependencies. it does not require
the model creator to use Pandas, numpy, or any other Python package to
transfer data to the model. This also means that it does not force the
user of the model to know any internal implementation details about the
model. On the other hand, Uber's solution requires that the user of the
model know how to work with Pandas dataframes. However, if the model
creator still wishes to accept numpy arrays or Pandas dataframes to
their model, the MLModel base class shown above still allows this.

By using python dictionaries for model input and output, the model is
easier to use. There is no need to understand how to use numpy arrays or
Pandas dataframes, remember the order of the columns, or know how the
output columns areencoded in order to use the model.

By stating the input and output schemas of a model programmatically, it
is possible to compare different model's schemas through automated
tools. This can be useful when tracking model changes across many
different versions of a model. Facebook's approach allows schema
metadata to be attached to ML models, but no other approach discussed
above does this.

By hiding the deserialization code behind the \_\_init\_\_() method, the
deserialization technique or the storage location of model files can be
changed without affecting the code that uses the model. In the same way,
I can replace the code in the predict() method without affecting the
user of the model, as long as the input and output schemas remain the
same. This is the benefit of using Object Oriented Programming to hide
implementation details from users of your code.

There are some other improvements that can be added to the MLModel base
class shown in this post, but these will be shown in a later blog post.
