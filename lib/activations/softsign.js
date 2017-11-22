'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = softsign;

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

const _softsign = (0, _cwise2.default)({
  args: ['array'],
  body: function (_x) {
    _x /= 1 + Math.abs(_x);
  }
});

/**
 * In-place operation: softsign activation function
 *
 * @param {Tensor} x
 */
function softsign(x) {
  _softsign(x.tensor);
}