'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = softplus;

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

const _softplus = (0, _cwise2.default)({
  args: ['array'],
  body: function (_x) {
    _x = Math.log(Math.exp(_x) + 1);
  }
});

/**
 * In-place operation: softplus activation function
 *
 * @param {Tensor} x
 */
function softplus(x) {
  _softplus(x.tensor);
}