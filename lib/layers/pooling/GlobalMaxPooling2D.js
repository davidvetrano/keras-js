'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _GlobalPooling2D2 = require('./_GlobalPooling2D');

var _GlobalPooling2D3 = _interopRequireDefault(_GlobalPooling2D2);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * GlobalMaxPooling2D layer class, extends abstract _GlobalPooling2D class
 */
class GlobalMaxPooling2D extends _GlobalPooling2D3.default {
  /**
   * Creates a GlobalMaxPooling2D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'GlobalMaxPooling2D';

    this.poolingFunc = 'max';
  }
}
exports.default = GlobalMaxPooling2D;