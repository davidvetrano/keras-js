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

var _tensorUtils = require('../../utils/tensorUtils');

var tensorUtils = _interopRequireWildcard(_tensorUtils);

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

var _ndarrayGemm = require('ndarray-gemm');

var _ndarrayGemm2 = _interopRequireDefault(_ndarrayGemm);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Conv2D layer class
 */
class Conv2D extends _Layer2.default {
  /**
   * Creates a Conv2D layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.filters] - Number of convolution filters to use
   * @param {number|number[]} [attrs.kernel_size] - Size of the convolution kernel
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Conv2D';

    const {
      filters = 1,
      kernel_size = [3, 3],
      strides = [1, 1],
      padding = 'valid',
      data_format = 'channels_last',
      dilation_rate = [1, 1],
      activation = 'linear',
      use_bias = true
    } = attrs;

    if (Array.isArray(kernel_size)) {
      this.kernelShape = [filters, ...kernel_size];
    } else {
      this.kernelShape = [filters, kernel_size, kernel_size];
    }

    if (Array.isArray(strides)) {
      this.strides = strides;
    } else {
      this.strides = [strides, strides];
    }

    if (padding === 'valid' || padding === 'same') {
      this.padding = padding;
    } else {
      this.throwError('Invalid padding.');
    }

    if (data_format === 'channels_last' || data_format === 'channels_first') {
      this.dataFormat = data_format;
    } else {
      this.throwError('Only channels_last and channels_first data formats are allowed.');
    }

    if (Array.isArray(dilation_rate)) {
      this.dilationRate = dilation_rate;
    } else {
      this.dilationRate = [dilation_rate, dilation_rate];
    }
    if ((this.dilationRate[0] !== 1 || this.dilationRate[1] !== 1) && (this.strides[0] !== 1 || this.strides[1] !== 1)) {
      // Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1
      // https://keras.io/layers/convolutional/#conv2d
      this.throwError(`Incompatible combination of dilation_rate with strides.`);
    }

    this.activation = activation;
    this.activationFunc = activations[activation];

    this.use_bias = use_bias;

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'bias'] : ['kernel'];

    // GPU setup
    if (this.gpu) {
      this.mapInputProgram = _WebGL.webgl2.compileProgram(require('../../mapInput.glsl'));
      this.matMulProgram = _WebGL.webgl2.compileProgram(require('../../matMul.glsl'));
      this.activationProgram = _WebGL.webgl2.compileProgram(require(`../../activations/${this.activation}.glsl`));
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   *
   * W weight tensor is converted to `channels_last` mode if in `channels_first` mode.
   *
   * In `channels_last` mode, W weight tensor has shape [nbRow, nbCol, inputChannels, nbFilter]
   *
   * In `channels_first` mode, W weight tensor has shape [nbFilter, inputChannels, nbRow, nbCol]
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    if (this.dataFormat === 'channels_first') {
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(2, 3, 1, 0);
    }
    super.setWeights(weightsArr, false);

    this._w2row();

    if (this.gpu) {
      this.weights['kernel'] = this.wRowsMat;
      this.weights['kernel'].createGLTexture();
      if (this.use_bias) {
        this.weights['bias'].createGLTexture();
      }
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

    const inputRows = inputShape[0];
    const inputCols = inputShape[1];
    const [nbFilter, nbRow, nbCol] = this.kernelShape;

    // effective shape after filter dilation
    const nbRowDilated = nbRow + (nbRow - 1) * (this.dilationRate[0] - 1);
    const nbColDilated = nbCol + (nbCol - 1) * (this.dilationRate[1] - 1);

    const outputRows = this.padding === 'same' ? Math.floor((inputRows + this.strides[0] - 1) / this.strides[0]) : Math.floor((inputRows - nbRowDilated + this.strides[0]) / this.strides[0]);
    const outputCols = this.padding === 'same' ? Math.floor((inputCols + this.strides[1] - 1) / this.strides[1]) : Math.floor((inputCols - nbColDilated + this.strides[1]) / this.strides[1]);
    const outputChannels = nbFilter;

    const paddingRow = this.padding === 'same' ? Math.max(0, Math.floor((outputRows - 1) * this.strides[0] + nbRowDilated - inputRows)) : 0;
    const paddingCol = this.padding === 'same' ? Math.max(0, Math.floor((outputCols - 1) * this.strides[1] + nbColDilated - inputCols)) : 0;
    const paddingRowBefore = Math.floor(paddingRow / 2);
    const paddingRowAfter = paddingRow - paddingRowBefore;
    const paddingColBefore = Math.floor(paddingCol / 2);
    const paddingColAfter = paddingCol - paddingColBefore;

    this.outputShape = [outputRows, outputCols, outputChannels];
    this.inputPadding = [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter];
  }

  /**
   * Pad input tensor if necessary, for padding='same'. See above for notes on calculating padding.
   *
   * @param {Tensor} x
   * @param {number} [padValue]
   * @returns {Tensor}
   */
  _padInput(x, padValue = 0) {
    if (this.padding === 'same') {
      const [inputRows, inputCols, inputChannels] = x.tensor.shape;
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding;
      const newRows = inputRows + paddingRowBefore + paddingRowAfter;
      const newCols = inputCols + paddingColBefore + paddingColAfter;
      let _x = new _Tensor2.default([], [newRows, newCols, inputChannels]);
      if (padValue !== 0) {
        _ndarrayOps2.default.assigns(_x.tensor, padValue);
      }
      _ndarrayOps2.default.assign(_x.tensor.hi(inputRows + paddingRowBefore, inputCols + paddingColBefore, inputChannels).lo(paddingRowBefore, paddingColBefore, 0), x.tensor);
      x.tensor = _x.tensor;
    }
    return x;
  }

  /**
   * Convert input tensor to column matrix
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  _im2col(x) {
    const [inputRows, inputCols, inputChannels] = x.tensor.shape;
    const nbRow = this.kernelShape[1];
    const nbCol = this.kernelShape[2];
    const outputRows = this.outputShape[0];
    const outputCols = this.outputShape[1];
    const nbPatches = outputRows * outputCols;
    const patchLen = nbRow * nbCol * inputChannels;

    // effective shape after filter dilation
    const nbRowDilated = nbRow + (nbRow - 1) * (this.dilationRate[0] - 1);
    const nbColDilated = nbCol + (nbCol - 1) * (this.dilationRate[1] - 1);

    if (!this.imColsMat) {
      this.imColsMat = new _Tensor2.default([], [nbPatches, patchLen]);
    }

    if (nbRowDilated === 1 && nbColDilated === 1 && this.strides[0] === 1 && this.strides[1] === 1) {
      this.imColsMat.replaceTensorData(x.tensor.data);
      if (this.gpu) {
        this.imColsMat.createGLTexture();
      }
      return this.imColsMat;
    }

    let patch = new _Tensor2.default([], [nbRow, nbCol, inputChannels]);
    let offset = 0;
    for (let i = 0, limit = inputRows - nbRowDilated; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputCols - nbColDilated; j <= limit; j += this.strides[1]) {
        _ndarrayOps2.default.assign(patch.tensor, x.tensor.hi(i + nbRowDilated, j + nbColDilated, inputChannels).lo(i, j, 0).step(this.dilationRate[0], this.dilationRate[1], 1));
        this.imColsMat.tensor.data.set(patch.tensor.data, offset);
        offset += patchLen;
      }
    }

    if (this.gpu) {
      this.imColsMat.createGLTexture();
    }
    return this.imColsMat;
  }

  /**
   * Convert filter weights to row matrix
   *
   * @returns {Tensor}
   */
  _w2row() {
    const inputChannels = this.weights['kernel'].tensor.shape[2];
    const [nbFilter, nbRow, nbCol] = this.kernelShape;
    const patchLen = nbRow * nbCol * inputChannels;

    this.wRowsMat = new _Tensor2.default([], [patchLen, nbFilter]);

    let patch = new _Tensor2.default([], [nbRow, nbCol, inputChannels]);
    let patchRaveled = new _Tensor2.default([], [patchLen]);
    for (let n = 0; n < nbFilter; n++) {
      _ndarrayOps2.default.assign(patch.tensor, this.weights['kernel'].tensor.pick(null, null, null, n));
      patchRaveled.replaceTensorData(patch.tensor.data);
      _ndarrayOps2.default.assign(this.wRowsMat.tensor.pick(null, n), patchRaveled.tensor);
    }

    return this.wRowsMat;
  }

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    this.inputShape = x.tensor.shape;
    this._calcOutputShape(this.inputShape);
    this._padInput(x);
    this._im2col(x);

    const nbFilter = this.kernelShape[0];
    const outputRows = this.outputShape[0];
    const outputCols = this.outputShape[1];
    const nbPatches = outputRows * outputCols;
    const matMul = new _Tensor2.default([], [nbPatches, nbFilter]);

    if (this.use_bias) {
      for (let n = 0; n < nbFilter; n++) {
        _ndarrayOps2.default.assigns(matMul.tensor.pick(null, n), this.weights['bias'].tensor.get(n));
      }
    }
    (0, _ndarrayGemm2.default)(matMul.tensor, this.imColsMat.tensor, this.wRowsMat.tensor, 1, 1);

    this.output = new _Tensor2.default([], this.outputShape);

    let outputChannelRaveled = new _Tensor2.default([], [outputRows * outputCols]);
    let outputChannel = new _Tensor2.default([], [outputRows, outputCols]);
    for (let n = 0; n < nbFilter; n++) {
      _ndarrayOps2.default.assign(outputChannelRaveled.tensor, matMul.tensor.pick(null, n));
      outputChannel.replaceTensorData(outputChannelRaveled.tensor.data);
      _ndarrayOps2.default.assign(this.output.tensor.pick(null, null, n), outputChannel.tensor);
    }

    this.activationFunc(this.output);

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      this.output.tensor = this.output.tensor.transpose(2, 0, 1);
    }
  }

  /**
   * Creates a index mapping from the 2D-reshaped input tensor with associated 3D tensor shape to the representation
   * required prior to the matrix multiply. This allows us to work directly on the 2D tensor representations rather
   * than needing to reshape to the 3D reprentation and calling im2col.
   *
   * @param {Object} indicesForReshaped
   */
  _createIndexMap(indicesForReshaped = {}) {
    if (this.rowIndexMap && this.colIndexMap) {
      return;
    }

    let [inputRows, inputCols, inputChannels] = this.inputShape;

    let indicesRow, indicesCol;
    if (indicesForReshaped.row && indicesForReshaped.col) {
      indicesRow = new _Tensor2.default(indicesForReshaped.row.data, indicesForReshaped.row.shape, { type: Int32Array });
      indicesCol = new _Tensor2.default(indicesForReshaped.col.data, indicesForReshaped.col.shape, { type: Int32Array });
    } else {
      indicesRow = new _Tensor2.default([], this.inputShape, { type: Int32Array });
      indicesCol = new _Tensor2.default([], this.inputShape, { type: Int32Array });
      for (let i = 0; i < inputRows; i++) {
        for (let j = 0; j < inputCols; j++) {
          _ndarrayOps2.default.assigns(indicesRow.tensor.pick(i, j, null), i * inputCols + j);
        }
      }
      for (let c = 0; c < inputChannels; c++) {
        _ndarrayOps2.default.assigns(indicesCol.tensor.pick(null, null, c), c);
      }
    }

    // padding for border mode 'same'
    if (this.padding === 'same') {
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding;
      inputRows = inputRows + paddingRowBefore + paddingRowAfter;
      inputCols = inputCols + paddingColBefore + paddingColAfter;
      const padValue = -1;
      this._padInput(indicesRow, padValue);
      this._padInput(indicesCol, padValue);
    }

    const nbRow = this.kernelShape[1];
    const nbCol = this.kernelShape[2];
    const outputRows = this.outputShape[0];
    const outputCols = this.outputShape[1];
    const nbPatches = outputRows * outputCols;
    const patchLen = nbRow * nbCol * inputChannels;

    // effective shape after filter dilation
    const nbRowDilated = nbRow + (nbRow - 1) * (this.dilationRate[0] - 1);
    const nbColDilated = nbCol + (nbCol - 1) * (this.dilationRate[1] - 1);

    this.rowIndexMap = new _Tensor2.default([], [nbPatches, patchLen], { type: Int32Array });
    this.colIndexMap = new _Tensor2.default([], [nbPatches, patchLen], { type: Int32Array });

    let indicesRowPatch = new _Tensor2.default([], [nbRow, nbCol, inputChannels]);
    let indicesColPatch = new _Tensor2.default([], [nbRow, nbCol, inputChannels]);
    let offset = 0;
    for (let i = 0, limit = inputRows - nbRowDilated; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputCols - nbColDilated; j <= limit; j += this.strides[1]) {
        _ndarrayOps2.default.assign(indicesRowPatch.tensor, indicesRow.tensor.hi(i + nbRowDilated, j + nbColDilated, inputChannels).lo(i, j, 0).step(this.dilationRate[0], this.dilationRate[1], 1));
        _ndarrayOps2.default.assign(indicesColPatch.tensor, indicesCol.tensor.hi(i + nbRowDilated, j + nbColDilated, inputChannels).lo(i, j, 0).step(this.dilationRate[0], this.dilationRate[1], 1));
        this.rowIndexMap.tensor.data.set(indicesRowPatch.tensor.data, offset);
        this.colIndexMap.tensor.data.set(indicesColPatch.tensor.data, offset);
        offset += patchLen;
      }
    }

    this.rowIndexMap.createGLTexture('2d', 'int');
    this.colIndexMap.createGLTexture('2d', 'int');
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (x.is2DReshaped) {
      this.inputShape = x.originalShape;
      this._calcOutputShape(this.inputShape);
      this._createIndexMap(x.indicesForReshaped);
      if (!this.mappedInput) {
        this.mappedInput = new _Tensor2.default([], this.rowIndexMap.glTextureShape);
        this.mappedInput.createGLTexture();
      }
    } else {
      this.inputShape = x.tensor.shape;
      this._calcOutputShape(this.inputShape);
      this._padInput(x);
      this._im2col(x);
    }

    // map from 2d-reshaped input
    if (x.is2DReshaped) {
      _WebGL.webgl2.runProgram({
        program: this.mapInputProgram,
        output: this.mappedInput,
        inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }, { texture: this.rowIndexMap.glTexture, type: '2d', name: 'rowIndexMap' }, { texture: this.colIndexMap.glTexture, type: '2d', name: 'colIndexMap' }]
      });
    }

    const input = x.is2DReshaped ? this.mappedInput : this.imColsMat;
    const outputTextureShape = [input.glTextureShape[0], this.weights['kernel'].glTextureShape[1]];

    // create output textures if doesn't already exist
    if (!this.outputPreactiv) {
      this.outputPreactiv = new _Tensor2.default([], outputTextureShape);
      this.outputPreactiv.createGLTexture();
      this.outputPreactiv.is2DReshaped = true;
      this.outputPreactiv.originalShape = this.outputShape;
      this.outputPreactiv.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.outputShape, false, -1);
    }
    if (!this.output) {
      this.output = new _Tensor2.default([], outputTextureShape);
      this.output.createGLTexture();
      this.output.is2DReshaped = true;
      this.output.originalShape = this.outputShape;
      this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.outputShape, false, -1);
    }

    // Matrix Multiply
    const matMulInputs = [{ texture: input.glTexture, type: '2d', name: 'A' }, { texture: this.weights['kernel'].glTexture, type: '2d', name: 'B' }];
    if (this.use_bias) {
      matMulInputs.push({ texture: this.weights['bias'].glTexture, type: '2d', name: 'C' });
    }
    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.outputPreactiv,
      inputs: matMulInputs,
      uniforms: [{ value: this.use_bias ? 1 : 0, type: 'bool', name: 'addC' }, { value: input.glTextureShape[0], type: 'int', name: 'M' }, { value: this.weights['kernel'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['kernel'].glTextureShape[1], type: 'int', name: 'N' }]
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
      this.output.reshapeFrom2D();

      // convert back to channels_first ordering if necessary
      if (this.dataFormat === 'channels_first') {
        this.output.tensor = this.output.tensor.transpose(2, 0, 1);
      }
    }
  }
}
exports.default = Conv2D;