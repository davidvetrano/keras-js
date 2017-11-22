'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = sigmoid;

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

const _sigmoid = (0, _cwise2.default)({
  args: ['array'],
  body: function (_x) {
    _x = 1 / (1 + Math.exp(-_x));
  }
});

/**
 * In-place operation: sigmoid activation function
 *
 * @param {Tensor} x
 */
function sigmoid(x) {
  _sigmoid(x.tensor);
}