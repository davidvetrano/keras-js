'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Pooling3D2 = require('./_Pooling3D');

var _Pooling3D3 = _interopRequireDefault(_Pooling3D2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * MaxPooling3D layer class, extends abstract _Pooling3D class
 */
class MaxPooling3D extends _Pooling3D3.default {
  /**
   * Creates a MaxPooling3D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'MaxPooling3D';

    this.poolingFunc = 'max';
  }
}
exports.default = MaxPooling3D;