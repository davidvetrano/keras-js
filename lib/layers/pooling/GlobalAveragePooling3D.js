'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _GlobalPooling3D2 = require('./_GlobalPooling3D');

var _GlobalPooling3D3 = _interopRequireDefault(_GlobalPooling3D2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * GlobalAveragePooling3D layer class, extends abstract _GlobalPooling3D class
 */
class GlobalAveragePooling3D extends _GlobalPooling3D3.default {
  /**
   * Creates a GlobalAveragePooling3D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'GlobalAveragePooling3D';

    this.poolingFunc = 'average';
  }
}
exports.default = GlobalAveragePooling3D;