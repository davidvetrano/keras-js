'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Flatten layer class
 * Turns tensor into 1-d. Note there is no concept of batch size in these layers (single-batch).
 */
class Flatten extends _Layer2.default {
  /**
   * Creates a Flatten layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Flatten';

    // GPU setup
    if (this.gpu) {
      this.program = _WebGL.webgl2.compileProgram(require('./Flatten.glsl'));
    }
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (this.gpu) {
      this._callGPU(x);
    } else {
      this._callCPU(x);
    }
    return this.output;
  }

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    if (x.tensor.shape.length <= 1) {
      this.output = x;
    } else {
      this.output = new _Tensor2.default([], [x.tensor.shape.reduce((a, b) => a * b, 1)]);
      this.output.replaceTensorData(x.tensor.data);
    }
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (!x.glTexture) {
      if (x.tensor.shape.length <= 2) {
        x.createGLTexture();
      } else if (x.tensor.shape.length > 2 && !x.is2DReshaped) {
        x.reshapeTo2D();
        x.createGLTexture();
      }
    }

    if (!this.output) {
      this.output = new _Tensor2.default([], [x.glTextureShape.reduce((a, b) => a * b, 1)]);
      this.output.createGLTexture();
    }

    _WebGL.webgl2.runProgram({
      program: this.program,
      output: this.output,
      inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }]
    });

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
    }
  }
}
exports.default = Flatten;