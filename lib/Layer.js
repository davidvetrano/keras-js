'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _WebGL = require('./WebGL2');

/**
 * Layer class
 */
class Layer {
  /**
   * Creates a layer
   *
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    this.layerClass = 'Layer';
    this.name = attrs.name;
    this.gpu = _WebGL.webgl2.isSupported && attrs.gpu;

    this.params = [];
    this.weights = {};

    this.inbound = [];
    this.outbound = [];
  }

  /**
   * Throws Error, adding layer context info to message
   *
   * @param {string} message
   */
  throwError(message) {
    throw new Error(`[${this.layerClass} layer: ${this.name || ''}] ${message}`);
  }

  /**
   * Set layer weights
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   * @param {boolean} createGLTexture
   */
  setWeights(weightsArr, createGLTexture = true) {
    this.params.forEach((p, i) => {
      this.weights[p] = weightsArr[i];

      if (this.gpu && createGLTexture) {
        this.weights[p].createGLTexture();
      }
    });
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    this.output = x;
    return this.output;
  }

  /**
   * Toggle GPU mode on/off
   *
   * @param {boolean} mode - on/off
   */
  toggleGPU(mode) {
    const newMode = typeof mode === 'undefined' ? !this.gpu : mode;
    if (_WebGL.webgl2.isSupported && newMode) {
      this.gpu = true;
    } else {
      this.gpu = false;
    }
  }
}
exports.default = Layer;