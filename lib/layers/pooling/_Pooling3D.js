'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _tensorUtils = require('../../utils/tensorUtils');

var tensorUtils = _interopRequireWildcard(_tensorUtils);

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * _Pooling3D layer class
 */
class _Pooling3D extends _Layer2.default {
  /**
   * Creates a _Pooling3D layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = '_Pooling3D';

    const { pool_size = [2, 2, 2], strides = null, padding = 'valid', data_format = 'channels_last' } = attrs;

    if (Array.isArray(pool_size)) {
      this.poolSize = pool_size;
    } else {
      this.poolSize = [pool_size, pool_size, pool_size];
    }

    if (Array.isArray(strides)) {
      this.strides = strides;
    } else if (strides !== null) {
      this.strides = [strides, strides, strides];
    } else {
      this.strides = this.poolSize;
    }

    this.padding = padding;
    this.dataFormat = data_format;

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
   * Method for computing output dimensions and padding, based on input dimensions, kernel size, and padding mode.
   *
   * For tensorflow implementation of padding, see:
   * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc
   *
   * @param {number[]} inputShape
   */
  _calcOutputShape(inputShape) {
    if (this.outputShape && this.inputPadding) {
      return;
    }

    const [inputDim1, inputDim2, inputDim3, inputChannels] = inputShape;
    const [poolDim1, poolDim2, poolDim3] = this.poolSize;

    const outputDim1 = this.padding === 'same' ? Math.floor((inputDim1 + this.strides[0] - 1) / this.strides[0]) : Math.floor((inputDim1 - poolDim1 + this.strides[0]) / this.strides[0]);
    const outputDim2 = this.padding === 'same' ? Math.floor((inputDim2 + this.strides[1] - 1) / this.strides[1]) : Math.floor((inputDim2 - poolDim2 + this.strides[1]) / this.strides[1]);
    const outputDim3 = this.padding === 'same' ? Math.floor((inputDim3 + this.strides[2] - 1) / this.strides[2]) : Math.floor((inputDim3 - poolDim3 + this.strides[2]) / this.strides[2]);

    const paddingDim1 = this.padding === 'same' ? Math.max(0, Math.floor((outputDim1 - 1) * this.strides[0] + poolDim1 - inputDim1)) : 0;
    const paddingDim2 = this.padding === 'same' ? Math.max(0, Math.floor((outputDim2 - 1) * this.strides[1] + poolDim2 - inputDim2)) : 0;
    const paddingDim3 = this.padding === 'same' ? Math.max(0, Math.floor((outputDim3 - 1) * this.strides[2] + poolDim3 - inputDim3)) : 0;
    const paddingDim1Before = Math.floor(paddingDim1 / 2);
    const paddingDim1After = paddingDim1 - paddingDim1Before;
    const paddingDim2Before = Math.floor(paddingDim2 / 2);
    const paddingDim2After = paddingDim2 - paddingDim2Before;
    const paddingDim3Before = Math.floor(paddingDim3 / 2);
    const paddingDim3After = paddingDim3 - paddingDim3Before;

    this.outputShape = [outputDim1, outputDim2, outputDim3, inputChannels];
    this.inputPadding = [paddingDim1Before, paddingDim1After, paddingDim2Before, paddingDim2After, paddingDim3Before, paddingDim3After];
  }

  /**
   * Pad input tensor if necessary, for padding='same'. See above for notes on calculating padding.
   *
   * For max, we pad with -infinity.
   *
   * For average we pad with zero.
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  _padInput(x) {
    if (this.padding === 'same') {
      const [inputDim1, inputDim2, inputDim3, inputChannels] = x.tensor.shape;
      const [paddingDim1Before, paddingDim1After, paddingDim2Before, paddingDim2After, paddingDim3Before, paddingDim3After] = this.inputPadding;
      const newDim1 = inputDim1 + paddingDim1Before + paddingDim1After;
      const newDim2 = inputDim2 + paddingDim2Before + paddingDim2After;
      const newDim3 = inputDim3 + paddingDim3Before + paddingDim3After;

      let _x = new _Tensor2.default([], [newDim1, newDim2, newDim3, inputChannels]);
      if (this.poolingFunc === 'max') {
        _ndarrayOps2.default.assigns(_x.tensor, Number.NEGATIVE_INFINITY);
      }

      _ndarrayOps2.default.assign(_x.tensor.hi(inputDim1 + paddingDim1Before, inputDim2 + paddingDim2Before, inputDim3 + paddingDim3Before, inputChannels).lo(paddingDim1Before, paddingDim2Before, paddingDim3Before, 0), x.tensor);
      x.tensor = _x.tensor;
    }
    return x;
  }

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    // convert to channels_last ordering
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 3, 0);
    }

    this._calcOutputShape(x.tensor.shape);
    this._padInput(x);

    const [inputDim1, inputDim2, inputDim3, inputChannels] = x.tensor.shape;
    const [poolDim1, poolDim2, poolDim3] = this.poolSize;
    this.output = new _Tensor2.default([], this.outputShape);
    let patch = new _Tensor2.default([], [poolDim1, poolDim2, poolDim3, inputChannels]);

    // keep track of padding since these values are not included in pooling
    // for max, we can ignore since padding values are set to -infinity
    const [paddingDim1Before, paddingDim1After, paddingDim2Before, paddingDim2After, paddingDim3Before, paddingDim3After] = this.inputPadding;

    for (let i = 0, _i = 0; i <= inputDim1 - poolDim1; i += this.strides[0], _i++) {
      let dim1InPadding = 0;
      if (i < paddingDim1Before) {
        dim1InPadding = paddingDim1Before - i;
      } else if (i + poolDim1 > inputDim1 - paddingDim1After) {
        dim1InPadding = i + poolDim1 - (inputDim1 - paddingDim1After);
      }

      for (let j = 0, _j = 0; j <= inputDim2 - poolDim2; j += this.strides[1], _j++) {
        let dim2InPadding = 0;
        if (j < paddingDim2Before) {
          dim2InPadding = paddingDim2Before - j;
        } else if (j + poolDim2 > inputDim2 - paddingDim2After) {
          dim2InPadding = j + poolDim2 - (inputDim2 - paddingDim2After);
        }

        for (let k = 0, _k = 0; k <= inputDim3 - poolDim3; k += this.strides[2], _k++) {
          let dim3InPadding = 0;
          if (k < paddingDim3Before) {
            dim3InPadding = paddingDim3Before - k;
          } else if (k + poolDim3 > inputDim3 - paddingDim3After) {
            dim3InPadding = k + poolDim3 - (inputDim3 - paddingDim3After);
          }
          const nbCellsEffective = (poolDim1 - dim1InPadding) * (poolDim2 - dim2InPadding) * (poolDim3 - dim3InPadding);

          _ndarrayOps2.default.assign(patch.tensor, x.tensor.hi(i + poolDim1, j + poolDim2, k + poolDim3, inputChannels).lo(i, j, k, 0));
          for (let c = 0; c < inputChannels; c++) {
            if (this.poolingFunc === 'max') {
              this.output.tensor.set(_i, _j, _k, c, _ndarrayOps2.default.sup(patch.tensor.pick(null, null, null, c)));
            } else if (this.poolingFunc === 'average') {
              this.output.tensor.set(_i, _j, _k, c, _ndarrayOps2.default.sum(patch.tensor.pick(null, null, null, c)) / nbCellsEffective);
            }
          }
        }
      }
    }

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      this.output.tensor = this.output.tensor.transpose(3, 0, 1, 2);
    }
  }

  /**
   * Convert input tensor to column matrix
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  _vol2col(x) {
    const [inputDim1, inputDim2, inputDim3, inputChannels] = x.tensor.shape;
    if (!this.tiledInput) {
      this.tiledInput = new _Tensor2.default([], [inputDim1 * inputDim2 * inputDim3, inputChannels]);
    }

    const patch = new _Tensor2.default([], [inputDim1, inputDim2, inputDim3]);
    const patchRaveled = new _Tensor2.default([], [inputDim1 * inputDim2 * inputDim3]);
    for (let c = 0; c < inputChannels; c++) {
      _ndarrayOps2.default.assign(patch.tensor, x.tensor.pick(null, null, null, c));
      patchRaveled.replaceTensorData(patch.tensor.data);
      _ndarrayOps2.default.assign(this.tiledInput.tensor.pick(null, c), patchRaveled.tensor);
    }

    if (this.gpu) {
      this.tiledInput.createGLTexture();
    }
    return this.tiledInput;
  }

  /**
   * Pre-compute index map for GPU pooling function
   */
  _createIndexMap() {
    if (this.indexMap) {
      return;
    }

    let inputDim1 = this.inputShape[0];
    let inputDim2 = this.inputShape[1];
    let inputDim3 = this.inputShape[2];
    const rowIndices = new _Tensor2.default([], [inputDim1, inputDim2, inputDim3]);
    let index = 0;
    for (let i = 0; i < inputDim1; i++) {
      for (let j = 0; j < inputDim2; j++) {
        for (let k = 0; k < inputDim3; k++) {
          rowIndices.tensor.set(i, j, k, index);
          index += 1;
        }
      }
    }

    // padding for border mode 'same'
    if (this.padding === 'same') {
      const [paddingDim1Before, paddingDim1After, paddingDim2Before, paddingDim2After, paddingDim3Before, paddingDim3After] = this.inputPadding;
      inputDim1 = inputDim1 + paddingDim1Before + paddingDim1After;
      inputDim2 = inputDim2 + paddingDim2Before + paddingDim2After;
      inputDim3 = inputDim3 + paddingDim3Before + paddingDim3After;
      const _rowIndices = new _Tensor2.default([], [inputDim1, inputDim2, inputDim3]);
      _ndarrayOps2.default.assigns(_rowIndices.tensor, -1);
      _ndarrayOps2.default.assign(_rowIndices.tensor.hi(this.inputShape[0] + paddingDim1Before, this.inputShape[1] + paddingDim2Before, this.inputShape[2] + paddingDim3Before).lo(paddingDim1Before, paddingDim2Before, paddingDim3Before), rowIndices.tensor);
      rowIndices.tensor = _rowIndices.tensor;
    }

    const [poolDim1, poolDim2, poolDim3] = this.poolSize;
    const outputDim1 = this.outputShape[0];
    const outputDim2 = this.outputShape[1];
    const outputDim3 = this.outputShape[2];

    this.indexMap = new _Tensor2.default([], [outputDim1 * outputDim2 * outputDim3, poolDim1 * poolDim2 * poolDim3], {
      type: Int32Array
    });

    let patchRow = new _Tensor2.default([], [poolDim1, poolDim2, poolDim3]);
    let offset = 0;
    for (let i = 0, limit = inputDim1 - poolDim1; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputDim2 - poolDim2; j <= limit; j += this.strides[1]) {
        for (let k = 0, limit = inputDim3 - poolDim3; k <= limit; k += this.strides[2]) {
          _ndarrayOps2.default.assign(patchRow.tensor, rowIndices.tensor.hi(i + poolDim1, j + poolDim2, k + poolDim3).lo(i, j, k));
          this.indexMap.tensor.data.set(patchRow.tensor.data, offset);
          offset += poolDim1 * poolDim2 * poolDim3;
        }
      }
    }

    this.indexMap.createGLTexture('2d', 'int');
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (x.is2DReshaped) {
      this.inputShape = x.originalShape;
    } else {
      // convert to channels_last ordering
      if (this.dataFormat === 'channels_first') {
        x.tensor = x.tensor.transpose(1, 2, 3, 0);
      }
      this.inputShape = x.tensor.shape;
      this._vol2col(x);
      x.glTexture = this.tiledInput.glTexture;
      x.glTextureShape = this.tiledInput.glTextureShape;
    }
    this._calcOutputShape(this.inputShape);
    this._createIndexMap();

    // create output textures if doesn't already exist
    if (!this.output) {
      const [outputDim1, outputDim2, outputDim3, inputChannels] = this.outputShape;
      const outputTextureShape = [outputDim1 * outputDim2 * outputDim3, inputChannels];
      this.output = new _Tensor2.default([], outputTextureShape);
      this.output.createGLTexture();
      this.output.is2DReshaped = true;
      this.output.originalShape = this.outputShape;
      this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.outputShape, false, -1);
    }

    const poolSize = this.poolSize[0] * this.poolSize[1] * this.poolSize[2];
    // `true` if max pooling, `false` if average pooling
    const isMaxPooling = this.poolingFunc === 'max';

    _WebGL.webgl2.runProgram({
      program: this.poolingProgram,
      output: this.output,
      inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }, { texture: this.indexMap.glTexture, type: '2d', name: 'indexMap' }],
      uniforms: [{ value: this.output.glTextureShape[0], type: 'int', name: 'rows' }, { value: this.output.glTextureShape[1], type: 'int', name: 'cols' }, { value: poolSize, type: 'int', name: 'poolSize' }, { value: +isMaxPooling, type: 'bool', name: 'isMaxPooling' }]
    });

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
      this.output.reshapeFrom2D();

      // convert back to channels_first ordering if necessary
      if (this.dataFormat === 'channels_first') {
        this.output.tensor = this.output.tensor.transpose(3, 0, 1, 2);
      }
    }
  }
}
exports.default = _Pooling3D;