'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _ndarrayUnsqueeze = require('ndarray-unsqueeze');

var _ndarrayUnsqueeze2 = _interopRequireDefault(_ndarrayUnsqueeze);

var _ndarrayTile = require('ndarray-tile');

var _ndarrayTile2 = _interopRequireDefault(_ndarrayTile);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * RepeatVector layer class
 * Turns 2D tensors of shape [features] to 3D tensors of shape [n, features].
 * Note there is no concept of batch size in these layers (single-batch) so we're actually going from 1D to 2D.
 */
class RepeatVector extends _Layer2.default {
  /**
   * Creates a RepeatVector layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.n]
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'RepeatVector';

    const { n = 1 } = attrs;
    this.n = n;

    // GPU setup
    if (this.gpu) {
      this.program = _WebGL.webgl2.compileProgram(require('./RepeatVector.glsl'));
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
    if (x.tensor.shape.length !== 1) {
      this.throwError('Only 1D tensor inputs allowed.');
    }
    this.output = new _Tensor2.default([], [this.n, x.tensor.shape[1]]);
    this.output.tensor = (0, _ndarrayTile2.default)((0, _ndarrayUnsqueeze2.default)(x.tensor, 0), [this.n, 1]);
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (!x.glTexture) {
      x.createGLTexture();
    }

    if (!this.output) {
      this.output = new _Tensor2.default([], [this.n, x.glTextureShape[1]]);
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
exports.default = RepeatVector;