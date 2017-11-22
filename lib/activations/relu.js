'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = relu;

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

var _Tensor = require('../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * In-place operation: ReLU activation function
 *
 * @param {Tensor} x
 * @param {{alpha: number, maxValue: number}} [opts]
 */
function relu(x, opts = {}) {
  const { alpha = 0, maxValue = null } = opts;
  let neg;
  if (alpha !== 0) {
    neg = new _Tensor2.default([], x.tensor.shape);
    _ndarrayOps2.default.mins(neg.tensor, x.tensor, 0);
    _ndarrayOps2.default.mulseq(neg.tensor, alpha);
  }
  _ndarrayOps2.default.maxseq(x.tensor, 0);
  if (maxValue) {
    _ndarrayOps2.default.minseq(x.tensor, maxValue);
  }
  if (neg) {
    _ndarrayOps2.default.addeq(x.tensor, neg.tensor);
  }
}