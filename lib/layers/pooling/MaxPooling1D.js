'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Pooling1D2 = require('./_Pooling1D');

var _Pooling1D3 = _interopRequireDefault(_Pooling1D2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * MaxPooling1D layer class, extends abstract _Pooling1D class
 */
class MaxPooling1D extends _Pooling1D3.default {
  /**
   * Creates a MaxPooling1D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'MaxPooling1D';

    this.poolingFunc = 'max';
  }
}
exports.default = MaxPooling1D;