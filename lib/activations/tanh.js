'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = tanh;

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

const _tanh = (0, _cwise2.default)({
  args: ['array'],
  body: function (_x) {
    _x = Math.tanh(_x);
  }
});

/**
 * In-place operation: tanh activation function
 *
 * @param {Tensor} x
 */
function tanh(x) {
  _tanh(x.tensor);
}