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
 * _GlobalPooling1D layer class
 */
class _GlobalPooling1D extends _Layer2.default {
  /**
   * Creates a _GlobalPooling1D layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = '_GlobalPooling1D';

    const { data_format = 'channels_last' } = attrs;
    this.dataFormat = data_format;

    // default pooling function
    // can be `max` or `average`
    this.poolingFunc = 'max';

    // GPU setup
    if (this.gpu) {
      this.poolingProgram = _WebGL.webgl2.compileProgram(require('./_GlobalPooling.glsl'));
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
    const [steps, features] = x.tensor.shape;
    this.output = new _Tensor2.default([], [features]);
    for (let i = 0, len = features; i < len; i++) {
      if (this.poolingFunc === 'max') {
        this.output.tensor.set(i, _ndarrayOps2.default.sup(x.tensor.pick(null, i)));
      } else if (this.poolingFunc === 'average') {
        this.output.tensor.set(i, _ndarrayOps2.default.sum(x.tensor.pick(null, i)) / steps);
      }
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
    this.inputShape = x.tensor.shape;

    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new _Tensor2.default([], [this.inputShape[1]]);
      this.output.createGLTexture();
    }

    // `true` if max pooling, `false` if average pooling
    const isMaxPooling = this.poolingFunc === 'max';

    _WebGL.webgl2.runProgram({
      program: this.poolingProgram,
      output: this.output,
      inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }],
      uniforms: [{ value: this.inputShape[0], type: 'int', name: 'channelDataSize' }, { value: +isMaxPooling, type: 'bool', name: 'isMaxPooling' }]
    });

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
    }
  }
}
exports.default = _GlobalPooling1D;