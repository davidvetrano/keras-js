'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.approxEquals = approxEquals;

var _ndarrayUnpack = require('ndarray-unpack');

var _ndarrayUnpack2 = _interopRequireDefault(_ndarrayUnpack);

var _flattenDeep = require('lodash/flattenDeep');

var _flattenDeep2 = _interopRequireDefault(_flattenDeep);

var _isFinite = require('lodash/isFinite');

var _isFinite2 = _interopRequireDefault(_isFinite);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Compares an ndarray's data element-wise to dataExpected,
 * within a certain tolerance. We unpack the ndarray first since
 * stride/offset prevents us from comparing the array data
 * element-wise directly.
 *
 * @param {Object} ndarrayOut
 * @param {number[]} dataExpected
 * @param {number} tol
 */
function approxEquals(ndarrayOut, dataExpected, tol = 0.0001) {
  const a = (0, _flattenDeep2.default)((0, _ndarrayUnpack2.default)(ndarrayOut));
  const b = dataExpected;
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (!(0, _isFinite2.default)(a[i])) {
      return false;
    }
    if (a[i] < b[i] - tol || a[i] > b[i] + tol) {
      return false;
    }
  }
  return true;
}