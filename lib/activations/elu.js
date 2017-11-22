'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = elu;

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

const _elu = (0, _cwise2.default)({
  args: ['array', 'scalar'],
  body: function (_x, alpha) {
    _x = Math.max(_x, 0) + alpha * (Math.exp(Math.min(_x, 0)) - 1);
  }
});

/**
 * In-place operation: ELU activation function
 *
 * @param {Tensor} x
 * @param {{alpha: number}} [opts]
 */
function elu(x, opts = {}) {
  const { alpha = 1.0 } = opts;
  _elu(x.tensor, alpha);
}