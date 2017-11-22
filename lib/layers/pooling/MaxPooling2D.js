'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Pooling2D2 = require('./_Pooling2D');

var _Pooling2D3 = _interopRequireDefault(_Pooling2D2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * MaxPooling2D layer class, extends abstract _Pooling2D class
 */
class MaxPooling2D extends _Pooling2D3.default {
  /**
   * Creates a MaxPooling2D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'MaxPooling2D';

    this.poolingFunc = 'max';
  }
}
exports.default = MaxPooling2D;