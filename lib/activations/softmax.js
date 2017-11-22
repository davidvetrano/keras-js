'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = softmax;

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * In-place operation: softmax activation function
 *
 * @param {Tensor} x
 */
function softmax(x) {
  if (x.tensor.shape.length === 1) {
    const maxval = _ndarrayOps2.default.sup(x.tensor);
    _ndarrayOps2.default.subseq(x.tensor, maxval);
    _ndarrayOps2.default.expeq(x.tensor);
    const sum = _ndarrayOps2.default.sum(x.tensor);
    _ndarrayOps2.default.divseq(x.tensor, sum);
  } else if (x.tensor.shape.length === 2) {
    for (let i = 0; i < x.tensor.shape[0]; i++) {
      const maxval = _ndarrayOps2.default.sup(x.tensor.pick(i, null));
      _ndarrayOps2.default.subseq(x.tensor.pick(i, null), maxval);
      _ndarrayOps2.default.expeq(x.tensor.pick(i, null));
      const sum = _ndarrayOps2.default.sum(x.tensor.pick(i, null));
      _ndarrayOps2.default.divseq(x.tensor.pick(i, null), sum);
    }
  } else {
    throw new Error(`[activations.softmax] tensor shape ${x.tensor.shape} not supported.`);
  }
}