'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _GlobalPooling3D2 = require('./_GlobalPooling3D');

var _GlobalPooling3D3 = _interopRequireDefault(_GlobalPooling3D2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * GlobalMaxPooling3D layer class, extends abstract _GlobalPooling3D class
 */
class GlobalMaxPooling3D extends _GlobalPooling3D3.default {
  /**
   * Creates a GlobalMaxPooling3D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'GlobalMaxPooling3D';

    this.poolingFunc = 'max';
  }
}
exports.default = GlobalMaxPooling3D;