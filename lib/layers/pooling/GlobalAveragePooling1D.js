'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _GlobalPooling1D2 = require('./_GlobalPooling1D');

var _GlobalPooling1D3 = _interopRequireDefault(_GlobalPooling1D2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * GlobalAveragePooling1D layer class, extends abstract _GlobalPooling1D class
 */
class GlobalAveragePooling1D extends _GlobalPooling1D3.default {
  /**
   * Creates a GlobalAveragePooling1D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'GlobalAveragePooling1D';

    this.poolingFunc = 'average';
  }
}
exports.default = GlobalAveragePooling1D;