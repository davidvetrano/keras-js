'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = hard_sigmoid;

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

// Reference hard sigmoid with slope and shift values from theano, see
// https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py
const _hard_sigmoid = (0, _cwise2.default)({
  args: ['array'],
  body: function (_x) {
    _x = _x * 0.2 + 0.5;
    if (_x <= 0) {
      _x = 0;
    } else if (_x >= 1) {
      _x = 1;
    }
  }
});

/**
 * In-place operation: hard sigmoid activation function
 *
 * @param {Tensor} x
 */
function hard_sigmoid(x) {
  _hard_sigmoid(x.tensor);
}