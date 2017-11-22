'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Pooling1D2 = require('./_Pooling1D');

var _Pooling1D3 = _interopRequireDefault(_Pooling1D2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * AveragePooling1D layer class, extends abstract _Pooling1D class
 */
class AveragePooling1D extends _Pooling1D3.default {
  /**
   * Creates a AveragePooling1D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'AveragePooling1D';

    this.poolingFunc = 'average';
  }
}
exports.default = AveragePooling1D;