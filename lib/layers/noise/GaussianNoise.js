'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * GaussianNoise layer class
 * Note that this layer is here only for compatibility purposes,
 * as it's only active during training phase.
 */
class GaussianNoise extends _Layer2.default {
  /**
   * Creates a GaussianNoise layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.stddev] - standard deviation of the noise distribution
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'GaussianNoise';

    const { stddev = 0 } = attrs;
    this.stddev = stddev;
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
exports.default = GaussianNoise;