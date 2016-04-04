#!/usr/bin/env python

import logging
import numpy
import sys
import os
import importlib
import cPickle

import theano
from theano import tensor

from blocks.extensions import Printing, SimpleExtension, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent
from predict import PredictDataStream

try:
    from blocks.extras.extensions.plot import Plot
    plot_avail = True
except ImportError:
    plot_avail = False
    print "No plotting extension available."

import data
from paramsaveload import SaveLoadParams

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

sys.setrecursionlimit(500000)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module('.%s' % model_name, 'config')

    # Build datastream
    path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/training")
    valid_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/validation")
    test_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/test")
    vocab_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/stats/training/vocab.txt")

    ds, train_stream = data.setup_datastream(path, vocab_path, config)
    # _, valid_stream = data.setup_datastream(valid_path, vocab_path, config)
    _, test_stream = data.setup_datastream(test_path, vocab_path, config)

    dump_path = os.path.join("model_params", model_name+".pkl")

    # Build model
    m = config.Model(config, ds.vocab_size)
    # Build the Blocks stuff for training
    model = Model(m.probs)#sgd_cost)
    # model = Model(m.sgd_cost)
    # print [v for l in m.monitor_vars for v in l]

    print "new model:---------------------------------------------------------"
    #print model.get_parameter_values()
    try:
        with open(dump_path, 'r') as f:
            logger.info('Loading parameters from %s...'%dump_path)
            model.set_parameter_values(cPickle.load(f))
    except IOError:
        pass

    #print model.get_parameter_dict()
    print "loaded model:---------------------------------------------------------"
    # tensor.nnet.softmax(m.sgd_cost)
    print model.outputs
    # print model.error_rate
    print "output ^ -------------\n"
    #print m.monitor_vars_valid
    # print model.error_rate
    print "mon vars valid ^ -------------\n"
    m.probs.name = "probs"
    m.cost.name = "cost"
    testpred = PredictDataStream(test_stream, [m.probs, m.cost] , "results.txt")
    testpred.do(None)
    print testpred.prediction

            #algorithm.
                #print model.get_parameter_values()

            #testpred = PredictDataStream(test_stream, model)

    # theano_function = model.get_theano_function([], model.monitor_vars)
    #
    # """Evaluate tensors for the given datastream."""
    # # predictions = OrderedDict([(var.name, []) for var in self.variables])
    # for batch in test_stream.get_epoch_iterator(as_dict=True):
    #     print batch
    #     print type(batch)
    #     keys = batch.keys()
    #     zipped = [dict(zip(keys, [[vals], [0]])) for vals in zip(*(batch[k] for k in keys))]
    # #    zipped = zip(batch)
    #     print zipped
    #     for subbatch in zipped:
    #         print "----"
    #         print subbatch
    #
    #         print type(subbatch)
    #         prediction = theano_function(**batch)
    #
    #
    #
    #         out = model.out[:, -1, :] / numpy.float32(sample_temperature)
    #     prob = tensor.nnet.softmax(out)
    #
    #     cg = ComputationGraph([prob])
    #     assert(len(cg.inputs) == 1)
    #     assert(cg.inputs[0].name == 'bytes')
    #
    #     # channel functions & state
    #     chfun = {}
    #     for ch in channels + ['']:
    #         logger.info("Building theano function for channel '%s'"%ch)
    #         state_vars = [theano.shared(v[0:1, :].zeros_like().eval(), v.name+'-'+ch)
    #                                 for v, _ in model.states]
    #         givens = [(v, x) for (v, _), x in zip(model.states, state_vars)]
    #         updates= [(x, upd) for x, (_, upd) in zip(state_vars, model.states)]
    #
    #         pred = theano.function(inputs=cg.inputs, outputs=[prob],
    #                                givens=givens, updates=updates)
    #         reset_states = theano.function(inputs=[], outputs=[],
    #                                        updates=[(v, v.zeros_like()) for v in state_vars
    #




        #    print prediction

        #for i, pred in zip(i, prediction):
        #    predictions[i].append(pred)
        #    print pred

    # accumulate predictions for the entire epoch
#    for var in self.variables:
#        predictions[var.name] = numpy.concatenate(predictions[var.name],
#                                                  axis=0)


    # func = model.get_theano_function()
    # #test_stream.get_data()
    # print "okay?"
    # #test_stream.open()
    # it = test_stream.get_epoch_iterator()#as_dict=True)
    # #for i, d in enumerate(it):
    # d = next(it)
    # print '--'
    # print d
    #
    # #    if i > 2: break
    # _1, outputs, _2, _3, costs = (func(d))
    # print outputs
#    for blah in test_stream
#        _1, outputs, _2, _3, costs = (func(input_))


#     return
#     # m =
#     # Build model
#     m = config.Model(config, ds.vocab_size)
#
#     # Build the Blocks stuff for training
#     model = Model(m.sgd_cost)
#
#     algorithm = GradientDescent(cost=m.sgd_cost,
#                                 step_rule=config.step_rule,
#                                 parameters=model.parameters)
#
#     extensions = [
#             TrainingDataMonitoring(
#                 [v for l in m.monitor_vars for v in l],
#                 prefix='train',
#                 every_n_batches=config.print_freq)
#     ]
#     #print [v for l in m.monitor_vars for v in l]
#     if config.save_freq is not None and dump_path is not None:
#         extensions += [
#             SaveLoadParams(path=dump_path,
#                            model=model,
#                            before_training=False,
#                            after_training=True,
#                            after_epoch=True,
#                            every_n_batches=config.save_freq)
#         ]
#
#
#     # if valid_stream is not None and config.valid_freq != -1:
#     #     extensions += [
#     #         DataStreamMonitoring(
#     #             [v for l in m.monitor_vars_valid for v in l],
#     #             valid_stream,
#     #             prefix='valid',
#     #             every_n_batches=config.valid_freq),
#     #     ]
#     if test_stream is not None:
#             extensions += [
#                 DataStreamMonitoring(
#                     [v for l in m.monitor_vars_valid for v in l],
#                     test_stream,
#                     prefix='test',
#                     every_n_batches=1), #config.valid_freq),
#             ]
#     if plot_avail:
#         plot_channels = [['train_' + v.name for v in lt] + ['valid_' + v.name for v in lv]
#                          for lt, lv in zip(m.monitor_vars, m.monitor_vars_valid)]
#         extensions += [
#             Plot(document='deepmind_qa_'+model_name,
#                  channels=plot_channels,
#                  # server_url='http://localhost:5006/', # If you need, change this
#                  every_n_batches=config.print_freq)
#         ]
#     extensions += [
#             Printing(every_n_batches=config.print_freq,
#                      after_epoch=True),
#             ProgressBar()
#     ]
#
#     main_loop = MainLoop(
#         model=model,
#         data_stream=train_stream,
#         algorithm=algorithm,
#         extensions=extensions
#     )
#
#     # Run the model !
#     #main_loop.run()
#     main_loop._check_finish_training()
#     main_loop.profile.report()
#
#
#
# #  vim: set sts=4 ts=4 sw=4 tw=0 et :
