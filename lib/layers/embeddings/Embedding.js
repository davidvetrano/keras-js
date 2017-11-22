'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Embedding layer class
 */
class Embedding extends _Layer2.default {
  /**
   * Creates a Embedding layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Embedding';

    const { input_dim = 1, output_dim = 1, input_length = 0, mask_zero = false } = attrs;

    this.inputDim = input_dim;
    this.outputDim = output_dim;
    this.inputLength = input_length;

    // mask_zero will be important for subsequent layers
    this.maskZero = mask_zero;

    // Layer weights specification
    this.params = ['embeddings'];

    // GPU setup
    if (this.gpu) {
      this.program = _WebGL.webgl2.compileProgram(require('./Embedding.glsl'));
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
    this.output = new _Tensor2.default([], [x.tensor.shape[0], this.weights['embeddings'].tensor.shape[1]]);

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      _ndarrayOps2.default.assign(this.output.tensor.pick(i, null), this.weights['embeddings'].tensor.pick(x.tensor.get(i), null));
    }
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
      this.output = new _Tensor2.default([], [x.glTextureShape[1], this.weights['embeddings'].glTextureShape[1]]);
      this.output.createGLTexture();
    }

    _WebGL.webgl2.runProgram({
      program: this.program,
      output: this.output,
      inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }, { texture: this.weights['embeddings'].glTexture, type: '2d', name: 'embeddings' }]
    });

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
    }
  }
}
exports.default = Embedding;