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
 * Conv2DTranspose layer class
 */
class Conv2DTranspose extends _Layer2.default {
  /**
   * Creates a Conv2DTranspose layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.filters] - Number of convolution filters to use.
   * @param {number|number[]} [attrs.kernel_size] - Size of the convolution kernel.
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Conv2DTranspose';

    const {
      filters = 1,
      kernel_size = [3, 3],
      strides = [1, 1],
      padding = 'valid',
      data_format = 'channels_last',
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

    this.activation = activation;
    this.activationFunc = activations[activation];

    this.use_bias = use_bias;

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'bias'] : ['kernel'];

    // GPU setup
    if (this.gpu) {
      this.matMulProgram = _WebGL.webgl2.compileProgram(require('../../matMul.glsl'));
      this.convTransposeProgram = _WebGL.webgl2.compileProgram(require('./Conv2DTranspose.glsl'));
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
   * For deconvolution, we will "take away" padding from the output rather than add padding to the input.
   *
   * For more details on calculating output shapes and padding for transposed convolutions (deconvolution here), see:
   * https://arxiv.org/pdf/1603.07285v1.pdf
   *
   * @param {number[]} inputShape
   */
  _calcOutputShape(inputShape) {
    if (this.outputShape && this.outputPadding) {
      return;
    }

    const inputRows = inputShape[0];
    const inputCols = inputShape[1];
    const [nbFilter, nbRow, nbCol] = this.kernelShape;

    const outputRows = this.padding === 'same' ? inputRows * this.strides[0] : inputRows * this.strides[0] + Math.max(nbRow - this.strides[0], 0);
    const outputCols = this.padding === 'same' ? inputCols * this.strides[1] : inputCols * this.strides[1] + Math.max(nbCol - this.strides[1], 0);
    const outputChannels = nbFilter;

    const paddingRow = this.padding === 'same' ? Math.max(0, Math.floor((inputRows - 1) * this.strides[0] + nbRow - outputRows)) : 0;
    const paddingCol = this.padding === 'same' ? Math.max(0, Math.floor((inputCols - 1) * this.strides[1] + nbCol - outputCols)) : 0;
    const paddingRowBefore = Math.floor(paddingRow / 2);
    const paddingRowAfter = paddingRow - paddingRowBefore;
    const paddingColBefore = Math.floor(paddingCol / 2);
    const paddingColAfter = paddingCol - paddingColBefore;

    this.outputShape = [outputRows, outputCols, outputChannels];
    this.outputPadding = [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter];
  }

  /**
   * Convert input image to column matrix, along channels axis
   *
   * shape: [inputRows, inputCols, inputChannels] -> [inputRows * inputCols, inputChannels]
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  _im2col(x) {
    const [inputRows, inputCols, inputChannels] = x.tensor.shape;

    if (!this.imColsMat) {
      this.imColsMat = new _Tensor2.default([], [inputRows * inputCols, inputChannels]);
    }

    let channelRaveled = new _Tensor2.default([], [inputRows * inputCols]);
    let channel = new _Tensor2.default([], [inputRows, inputCols]);
    for (let c = 0; c < inputChannels; c++) {
      _ndarrayOps2.default.assign(channel.tensor, x.tensor.pick(null, null, c));
      channelRaveled.replaceTensorData(channel.tensor.data);
      _ndarrayOps2.default.assign(this.imColsMat.tensor.pick(null, c), channelRaveled.tensor);
    }
    if (this.gpu) {
      this.imColsMat.createGLTexture();
    }
    return this.imColsMat;
  }

  /**
   * Convert filter weights to row matrix, along channels axis
   *
   * shape: [nbRow, nbCol, nbFilter, inputChannels] -> [inputChannels, nbRow * nbCol * nbFilter]
   *
   * @returns {Tensor}
   */
  _w2row() {
    const [nbRow, nbCol, nbFilter, inputChannels] = this.weights['kernel'].tensor.shape;

    this.wRowsMat = new _Tensor2.default([], [inputChannels, nbRow * nbCol * nbFilter]);

    let channelRaveled = new _Tensor2.default([], [nbRow * nbCol * nbFilter]);
    let channel = new _Tensor2.default([], [nbRow, nbCol, nbFilter]);
    for (let c = 0; c < inputChannels; c++) {
      _ndarrayOps2.default.assign(channel.tensor, this.weights['kernel'].tensor.pick(null, null, null, c));
      channelRaveled.replaceTensorData(channel.tensor.data);
      _ndarrayOps2.default.assign(this.wRowsMat.tensor.pick(c, null), channelRaveled.tensor);
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
    this._im2col(x);

    const inputRows = x.tensor.shape[0];
    const inputCols = x.tensor.shape[1];
    const [nbFilter, nbRow, nbCol] = this.kernelShape;
    const matMul = new _Tensor2.default([], [inputRows * inputCols, nbRow * nbCol * nbFilter]);

    (0, _ndarrayGemm2.default)(matMul.tensor, this.imColsMat.tensor, this.wRowsMat.tensor, 1, 1);

    // add padding which we will take away later
    const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.outputPadding;
    this.output = new _Tensor2.default([], this.outputShape);
    let outputPadded = new _Tensor2.default([], [this.outputShape[0] + paddingRowBefore + paddingRowAfter, this.outputShape[1] + paddingColBefore + paddingColAfter, this.outputShape[2]]);

    const patchShape = [nbRow, nbCol, nbFilter];
    let patch = new _Tensor2.default([], patchShape);
    let patchRaveled = new _Tensor2.default([], [nbRow * nbCol * nbFilter]);
    let index = 0;
    for (let i = 0; i < inputRows; i++) {
      for (let j = 0; j < inputCols; j++) {
        _ndarrayOps2.default.assign(patchRaveled.tensor, matMul.tensor.pick(index, null));
        patch.replaceTensorData(patchRaveled.tensor.data);
        const iOutPos = i * this.strides[0];
        const jOutPos = j * this.strides[1];
        _ndarrayOps2.default.addeq(outputPadded.tensor.hi(iOutPos + nbRow, jOutPos + nbCol, this.outputShape[2]).lo(iOutPos, jOutPos, 0), patch.tensor);
        index += 1;
      }
    }

    // remove padding
    _ndarrayOps2.default.assign(this.output.tensor, outputPadded.tensor.hi(this.outputShape[0] + paddingRowBefore, this.outputShape[1] + paddingColBefore, this.outputShape[2]).lo(paddingRowBefore, paddingColBefore, 0));

    // bias
    if (this.use_bias) {
      for (let n = 0; n < nbFilter; n++) {
        _ndarrayOps2.default.addseq(this.output.tensor.pick(null, null, n), this.weights['bias'].tensor.get(n));
      }
    }

    this.activationFunc(this.output);

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      this.output.tensor = this.output.tensor.transpose(2, 0, 1);
    }
  }

  /**
   * In GPU mode, we work directly on 2D-reshaped representations of the tensors. After the matrix multiply step produce
   * matrix Y, the final output Z at coordinate [i,j] will be the summation of a number of elements of the matrix Y.
   * Here, we calculate the indices of matrix Y for each coordinate [i,j] of Z, and encode these index maps as texture
   * arrays.
   */
  _createIndexMap() {
    if (this.rowIndexMap && this.colIndexMap) {
      return;
    }

    const inputRows = this.inputShape[0];
    const inputCols = this.inputShape[1];
    const [nbFilter, nbRow, nbCol] = this.kernelShape;

    const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.outputPadding;

    const indicesMapShape = [...this.outputShape, inputRows * inputCols];
    const indicesMapShapePadded = [this.outputShape[0] + paddingRowBefore + paddingRowAfter, this.outputShape[1] + paddingColBefore + paddingColAfter, this.outputShape[2], inputRows * inputCols];
    const outputRowIndicesMap = new _Tensor2.default([], indicesMapShape, { type: Int32Array });
    const outputColIndicesMap = new _Tensor2.default([], indicesMapShape, { type: Int32Array });
    const outputRowIndicesMapPadded = new _Tensor2.default([], indicesMapShapePadded, { type: Int32Array });
    const outputColIndicesMapPadded = new _Tensor2.default([], indicesMapShapePadded, { type: Int32Array });
    _ndarrayOps2.default.assigns(outputRowIndicesMap.tensor, -1);
    _ndarrayOps2.default.assigns(outputColIndicesMap.tensor, -1);
    _ndarrayOps2.default.assigns(outputRowIndicesMapPadded.tensor, -1);
    _ndarrayOps2.default.assigns(outputColIndicesMapPadded.tensor, -1);

    const matMulColIndicesPatch = new _Tensor2.default([], [nbRow, nbCol, nbFilter, 1], { type: Int32Array });
    for (let i = 0; i < nbRow * nbCol * nbFilter; i++) {
      matMulColIndicesPatch.tensor.data[i] = i;
    }

    for (let i = 0; i < inputRows; i++) {
      for (let j = 0; j < inputCols; j++) {
        const matMulRowIndex = i * inputCols + j;
        const iOutPos = i * this.strides[0];
        const jOutPos = j * this.strides[1];
        _ndarrayOps2.default.assigns(outputRowIndicesMapPadded.tensor.hi(iOutPos + nbRow, jOutPos + nbCol, this.outputShape[2], matMulRowIndex + 1).lo(iOutPos, jOutPos, 0, matMulRowIndex), matMulRowIndex);
        _ndarrayOps2.default.assign(outputColIndicesMapPadded.tensor.hi(iOutPos + nbRow, jOutPos + nbCol, this.outputShape[2], matMulRowIndex + 1).lo(iOutPos, jOutPos, 0, matMulRowIndex), matMulColIndicesPatch.tensor);
      }
    }

    // remove padding
    _ndarrayOps2.default.assign(outputRowIndicesMap.tensor, outputRowIndicesMapPadded.tensor.hi(this.outputShape[0] + paddingRowBefore, this.outputShape[1] + paddingColBefore, this.outputShape[2], inputRows * inputCols).lo(paddingRowBefore, paddingColBefore, 0, 0));
    _ndarrayOps2.default.assign(outputColIndicesMap.tensor, outputColIndicesMapPadded.tensor.hi(this.outputShape[0] + paddingRowBefore, this.outputShape[1] + paddingColBefore, this.outputShape[2], inputRows * inputCols).lo(paddingRowBefore, paddingColBefore, 0, 0));
    // combine first two dimensions
    const tiledIndicesMapShape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2], inputRows * inputCols];
    this.rowIndexMap = new _Tensor2.default([], tiledIndicesMapShape, { type: Int32Array });
    this.colIndexMap = new _Tensor2.default([], tiledIndicesMapShape, { type: Int32Array });
    const channelData = new _Tensor2.default([], [this.outputShape[2], inputRows * inputCols], { type: Int32Array });
    for (let i = 0; i < this.outputShape[0]; i++) {
      for (let j = 0; j < this.outputShape[1]; j++) {
        _ndarrayOps2.default.assign(channelData.tensor, outputRowIndicesMap.tensor.pick(i, j, null, null));
        _ndarrayOps2.default.assign(this.rowIndexMap.tensor.pick(i * this.outputShape[1] + j, null, null), channelData.tensor);
        _ndarrayOps2.default.assign(channelData.tensor, outputColIndicesMap.tensor.pick(i, j, null, null));
        _ndarrayOps2.default.assign(this.colIndexMap.tensor.pick(i * this.outputShape[1] + j, null, null), channelData.tensor);
      }
    }

    if (this.gpu) {
      this.rowIndexMap.createGLTexture('2d_array', 'int');
      this.colIndexMap.createGLTexture('2d_array', 'int');
    }
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
    } else {
      this.inputShape = x.tensor.shape;
      this._calcOutputShape(this.inputShape);
      this._im2col(x);
      x.glTexture = this.imColsMat.glTexture;
      x.glTextureShape = this.imColsMat.glTextureShape;
    }

    // create output textures if doesn't already exist
    if (!this.outputMatmul) {
      const outputTextureShape = [x.glTextureShape[0], this.weights['kernel'].glTextureShape[1]];
      this.outputMatmul = new _Tensor2.default([], outputTextureShape);
      this.outputMatmul.createGLTexture();
    }
    if (!this.outputPreactiv) {
      const outputTextureShape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]];
      this.outputPreactiv = new _Tensor2.default([], outputTextureShape);
      this.outputPreactiv.createGLTexture();
      this.outputPreactiv.is2DReshaped = true;
      this.outputPreactiv.originalShape = this.outputShape;
      this.outputPreactiv.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.outputShape, false, -1);
    }
    if (!this.output) {
      const outputTextureShape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]];
      this.output = new _Tensor2.default([], outputTextureShape);
      this.output.createGLTexture();
      this.output.is2DReshaped = true;
      this.output.originalShape = this.outputShape;
      this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.outputShape, false, -1);
    }

    // Matrix Multiply with kernel
    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.outputMatmul,
      inputs: [{ texture: x.glTexture, type: '2d', name: 'A' }, { texture: this.weights['kernel'].glTexture, type: '2d', name: 'B' }],
      uniforms: [{ value: 0, type: 'bool', name: 'addC' }, { value: x.glTextureShape[0], type: 'int', name: 'M' }, { value: this.weights['kernel'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['kernel'].glTextureShape[1], type: 'int', name: 'N' }]
    });

    // Tranposed Convolution
    this._createIndexMap();
    const test = new _Tensor2.default([], [this.outputShape[0] * this.outputShape[1], this.outputShape[2]]);
    _ndarrayOps2.default.assign(test.tensor, this.rowIndexMap.tensor.pick(null, null, 0));
    const convTransposeInputs = [{ texture: this.outputMatmul.glTexture, type: '2d', name: 'outputMatmul' }, { texture: this.rowIndexMap.glTexture, type: '2d_array', name: 'rowIndexMap' }, { texture: this.colIndexMap.glTexture, type: '2d_array', name: 'colIndexMap' }];
    if (this.use_bias) {
      convTransposeInputs.push({ texture: this.weights['bias'].glTexture, type: '2d', name: 'bias' });
    }
    _WebGL.webgl2.runProgram({
      program: this.convTransposeProgram,
      output: this.outputPreactiv,
      inputs: convTransposeInputs,
      uniforms: [{ value: this.use_bias ? 1 : 0, type: 'bool', name: 'use_bias' }, { value: this.outputShape[0] * this.outputShape[1], type: 'int', name: 'rows' }, { value: this.outputShape[2], type: 'int', name: 'cols' }, { value: this.inputShape[0] * this.inputShape[1], type: 'int', name: 'summationLength' }]
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
exports.default = Conv2DTranspose;