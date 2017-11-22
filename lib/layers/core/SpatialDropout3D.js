'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * SpatialDropout3D layer class
 * Note that this layer is here only for compatibility purposes,
 * as it's only active during training phase.
 */
class SpatialDropout3D extends _Layer2.default {
  /**
   * Creates an SpatialDropout3D layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.rate] - fraction of the input units to drop (between 0 and 1)
   * @param {string} [attrs.data_format] - channels_first` or `channels_last`
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'SpatialDropout3D';

    const { rate = 0.5, data_format = 'channels_last' } = attrs;

    this.rate = Math.min(Math.max(0, rate), 1);
    this.dataFormat = data_format;
  }

  /**
   * Method for layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    this.output = x;
    return this.output;
  }
}
exports.default = SpatialDropout3D;