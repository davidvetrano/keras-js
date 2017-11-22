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
 * _Pooling1D layer class
 */
class _Pooling1D extends _Layer2.default {
  /**
   * Creates a _Pooling1D layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = '_Pooling1D';

    const { pool_size = 2, strides = null, padding = 'valid' } = attrs;

    this.poolSize = pool_size;
    this.strides = strides === null ? this.poolSize : strides;
    this.padding = padding;

    // default pooling function
    // can be `max` or `average`
    this.poolingFunc = 'max';

    // GPU setup
    if (this.gpu) {
      this.poolingProgram = _WebGL.webgl2.compileProgram(require('./_Pooling.glsl'));
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
    const stepsNew = this.padding === 'valid' ? Math.floor((x.tensor.shape[0] - this.poolSize + this.strides) / this.strides) : Math.floor((x.tensor.shape[0] + this.strides - 1) / this.strides);

    this.output = new _Tensor2.default([], [stepsNew, x.tensor.shape[1]]);
    const outputStep = new _Tensor2.default([], [x.tensor.shape[1]]);

    // in padding same, start negative from beyond step 0
    let step = this.padding === 'valid' ? 0 : Math.min(0, Math.ceil((x.tensor.shape[0] - (stepsNew - 1) * this.strides - this.poolSize) / 2));

    for (let i = 0; i < stepsNew; i++) {
      let _step = Math.max(0, step);
      let limit = this.poolSize + Math.min(0, step);
      _ndarrayOps2.default.assign(outputStep.tensor, x.tensor.pick(_step, null));

      let count = 1;
      for (let j = 1; j < limit; j++) {
        if (_step + j > x.tensor.shape[0] - 1) {
          break;
        }
        if (this.poolingFunc === 'max') {
          _ndarrayOps2.default.maxeq(outputStep.tensor, x.tensor.pick(_step + j, null));
        } else if (this.poolingFunc === 'average') {
          _ndarrayOps2.default.addeq(outputStep.tensor, x.tensor.pick(_step + j, null));
        }
        count += 1;
      }

      if (this.poolingFunc === 'average') {
        _ndarrayOps2.default.divseq(outputStep.tensor, count);
      }

      _ndarrayOps2.default.assign(this.output.tensor.pick(i, null), outputStep.tensor);
      step += this.strides;
    }
  }

  /**
   * Pre-compute index map for GPU pooling function
   */
  _createIndexMap() {
    if (this.indexMap) {
      return;
    }

    const stepsNew = this.padding === 'valid' ? Math.floor((this.inputShape[0] - this.poolSize + this.strides) / this.strides) : Math.floor((this.inputShape[0] + this.strides - 1) / this.strides);

    this.outputShape = [stepsNew, this.inputShape[1]];

    this.indexMap = new _Tensor2.default([], [stepsNew, this.poolSize], { type: Int32Array });
    _ndarrayOps2.default.assigns(this.indexMap.tensor, -1);

    // in padding same, start negative from beyond step 0
    let step = this.padding === 'valid' ? 0 : Math.min(0, Math.ceil((this.inputShape[0] - (stepsNew - 1) * this.strides - this.poolSize) / 2));

    for (let i = 0; i < stepsNew; i++) {
      let _step = Math.max(0, step);
      let limit = this.poolSize + Math.min(0, step);

      let inputIndex = _step;
      this.indexMap.tensor.set(i, 0, inputIndex);
      for (let j = 1; j < limit; j++) {
        inputIndex = _step + j;
        if (inputIndex <= this.inputShape[0] - 1) {
          this.indexMap.tensor.set(i, j, inputIndex);
        } else {
          break;
        }
      }
      step += this.strides;
    }

    this.indexMap.createGLTexture('2d', 'int');
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
    this._createIndexMap();

    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new _Tensor2.default([], this.outputShape);
      this.output.createGLTexture();
    }

    // `true` if max pooling, `false` if average pooling
    const isMaxPooling = this.poolingFunc === 'max';

    _WebGL.webgl2.runProgram({
      program: this.poolingProgram,
      output: this.output,
      inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }, { texture: this.indexMap.glTexture, type: '2d', name: 'indexMap' }],
      uniforms: [{ value: this.output.glTextureShape[0], type: 'int', name: 'rows' }, { value: this.output.glTextureShape[1], type: 'int', name: 'cols' }, { value: this.poolSize, type: 'int', name: 'poolSize' }, { value: +isMaxPooling, type: 'bool', name: 'isMaxPooling' }]
    });

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
    }
  }
}
exports.default = _Pooling1D;