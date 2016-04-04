import theano
from theano import tensor
import numpy
from blocks.graph import ComputationGraph
# x = tensor.arange(10)*10
# s =  numpy.asarray([2, 8, 1], dtype=numpy.int32)
#
# # s =  numpy.asarray([2, 8, 1, 7, 7, 7, 7, 7, 7, 7], dtype=numpy.float32)
# # test = tensor.eq(x, s)
# # a = test.eval()
# # print a
# #xSub = theano.shared(x)
# a = x[s].eval()
# print a



vocab = tensor.arange(10)
probs = numpy.asarray([0, 0.8, 0, 0.2], dtype=numpy.float32)
context = numpy.asarray([3, 2, 8, 1], dtype=numpy.int32)
ans3 =  numpy.asarray([2, 8, 1], dtype=numpy.int32)
ans1 =  numpy.asarray([1, 3, 4], dtype=numpy.int32)
ans2 =  numpy.asarray([1, 1, 4], dtype=numpy.int32)
# s =  numpy.asarray([2, 8, 1, 7, 7, 7, 7, 7, 7, 7], dtype=numpy.float32)
# test = tensor.eq(x, s)
# a = test.eval()
# print a
#xSub = theano.shared(x)
probsPadded = tensor.zeros_like(vocab, dtype=numpy.float32)
probsSubset = probsPadded[context]
b = tensor.set_subtensor(probsSubset, probs)

ans1probs = b[ans1]
ans1score = ans1probs.sum()
ans2probs = b[ans2]
ans2score = ans2probs.sum()
ans3probs = b[ans3]
ans3score = ans3probs.sum()
allans = tensor.stacklists([ans1score, ans2score, ans3score])
pred = tensor.argmax(allans)


cg = ComputationGraph([ans1probs, ans1score, ans2probs, ans2score, ans3probs, ans3score, allans, pred])
f = cg.get_theano_function()
out = f()
# a = probsPadded.eval()
# be = b.eval()
# a1p = ans1probs.eval()
# a1 = ans1score.eval()
# print a
# print be
# print a1p
# print a1
print out
