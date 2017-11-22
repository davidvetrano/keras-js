'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Pooling2D2 = require('./_Pooling2D');

var _Pooling2D3 = _interopRequireDefault(_Pooling2D2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * AveragePooling2D layer class, extends abstract _Pooling2D class
 */
class AveragePooling2D extends _Pooling2D3.default {
  /**
   * Creates a AveragePooling2D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'AveragePooling2D';

    this.poolingFunc = 'average';
  }
}
exports.default = AveragePooling2D;