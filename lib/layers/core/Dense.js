'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _activations = require('../../activations');

var activations = _interopRequireWildcard(_activations);

var _WebGL = require('../../WebGL2');

var _ndarrayBlasLevel = require('ndarray-blas-level2');

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Dense layer class
 */
class Dense extends _Layer2.default {
  /**
   * Creates a Dense layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.units] - output dimension size
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Dense';

    const { units = 1, activation = 'linear', input_dim = null, use_bias = true } = attrs;

    this.activation = activation;
    this.activationFunc = activations[this.activation];
    this.units = units;
    this.input_dim = input_dim;
    this.use_bias = use_bias;

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'bias'] : ['kernel'];

    // Input shape specification
    if (this.input_dim) {
      this.inputShape = [this.input_dim];
    }

    // Output
    this.outputPreactiv = new _Tensor2.default([], [this.units]);
    this.output = new _Tensor2.default([], [this.units]);

    // GPU setup
    if (this.gpu) {
      this.matMulProgram = _WebGL.webgl2.compileProgram(require('../../matMul.glsl'));
      this.activationProgram = _WebGL.webgl2.compileProgram(require(`../../activations/${this.activation}.glsl`));
      this.outputPreactiv.createGLTexture();
      this.output.createGLTexture();
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
    if (this.use_bias) {
      _ndarrayOps2.default.assign(this.output.tensor, this.weights['bias'].tensor);
    }
    (0, _ndarrayBlasLevel.gemv)(1, this.weights['kernel'].tensor.transpose(1, 0), x.tensor, 1, this.output.tensor);
    this.activationFunc(this.output);
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

    // Matrix Multiply
    const matMulInputs = [{ texture: x.glTexture, type: '2d', name: 'A' }, { texture: this.weights['kernel'].glTexture, type: '2d', name: 'B' }];
    if (this.use_bias) {
      matMulInputs.push({ texture: this.weights['bias'].glTexture, type: '2d', name: 'C' });
    }
    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.outputPreactiv,
      inputs: matMulInputs,
      uniforms: [{ value: this.use_bias ? 1 : 0, type: 'bool', name: 'addC' }, { value: x.glTextureShape[0], type: 'int', name: 'M' }, { value: this.weights['kernel'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['kernel'].glTextureShape[1], type: 'int', name: 'N' }]
    });

    // Activation
    if (this.activation === 'linear') {
      this.output = this.outputPreactiv;
    } else {
      _WebGL.webgl2.runProgram({
        program: this.activationProgram,
        output: this.output,
        inputs: [{ texture: this.outputPreactiv.glTexture, type: '2d', name: 'x' }]
      });
    }

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
    }
  }
}
exports.default = Dense;