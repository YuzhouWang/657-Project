from collections import OrderedDict

import numpy

from blocks.extensions import SimpleExtension
from blocks.graph import ComputationGraph


class PredictDataStream(SimpleExtension):
    """Evaluate tensors for a given datastream.

    Parameters
    ----------
    data_stream : `~fuel.streams.DataStream`
        Data stream for which to evaluate.
    variables : list of `~theano.TensorVariable`
        The variables to evaluate.
    path : str, optional
        The destination path for pickling. If not given, the output
        will not be saved, but will be stored in the `prediction`
        attribute instead.

    Attributes
    ----------
    prediction : `collections.OrderedDict`
        Storage for tensor evaluations. Maps tensor names to outputs
        of type `numpy.ndarray`. If the callback has not been invoked,
        this attribute is set to None.

    """
    def __init__(self, data_stream, variables, path=None, **kwargs):
        self.data_stream = data_stream
        self.variables = variables
        # for zip(var, var1) in self.variables, variables
        #     var.name = var1.name
        #print (var.name for var in variables)
        #print "varnames ^"
        #self.variables.name = variables.name
        self.path = path
        self.prediction = None

        kwargs.setdefault('after_training', True)
        super(PredictDataStream, self).__init__(**kwargs)

        cg = ComputationGraph(variables)
        self.theano_function = cg.get_theano_function()

    def do(self, which_callback, *args):
        """Evaluate tensors for the given datastream."""
        print self.variables

        predictions = OrderedDict([(var.name, []) for var in self.variables])

        print "---------------------\n"
        print (var.name for var in self.variables)

        for batch in self.data_stream.get_epoch_iterator(as_dict=True):
            print batch
            print "first\n"
            prediction = self.theano_function(**batch)
            print prediction
            # theano.printing.debugprint(prediction)
            for var, pred in zip(self.variables, prediction):

                if var.name is None:
                    var.name = "blah"
                print "second: " + var.name + "\n"
                predictions[var.name].append(pred)

        print "third\n"
        print predictions
        # accumulate predictions for the entire epoch
        # for var in self.variables:
        #
        #     predictions[var.name] = numpy.concatenate(predictions[var.name],
        #                                               axis=0)
        if self.path is not None:
            numpy.savez(self.path, **predictions)
