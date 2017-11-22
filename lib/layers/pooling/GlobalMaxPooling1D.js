'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _GlobalPooling1D2 = require('./_GlobalPooling1D');

var _GlobalPooling1D3 = _interopRequireDefault(_GlobalPooling1D2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * GlobalMaxPooling1D layer class, extends abstract _GlobalPooling1D class
 */
class GlobalMaxPooling1D extends _GlobalPooling1D3.default {
  /**
   * Creates a GlobalMaxPooling1D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'GlobalMaxPooling1D';

    this.poolingFunc = 'max';
  }
}
exports.default = GlobalMaxPooling1D;