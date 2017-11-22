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

var _Conv2D = require('./Conv2D');

var _Conv2D2 = _interopRequireDefault(_Conv2D);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * _DepthwiseConv2D layer class
 */
class _DepthwiseConv2D extends _Conv2D2.default {
  /**
   * @param {Object} [attrs]
   */
  constructor(attrs = {}) {
    super(attrs);
  }

  /**
   * @param {number[]} inputShape
   */
  _calcOutputShape(inputShape) {
    super._calcOutputShape(inputShape);
    const nbFilter = this.kernelShape[0];
    const inputChannels = inputShape[2];
    this.outputShape[2] = nbFilter * inputChannels;
  }

  /**
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
    const patchLen = nbRow * nbCol;

    if (!this.imColsMat) {
      this.imColsMat = new _Tensor2.default([], [nbPatches * inputChannels, patchLen]);
    }

    let patch = new _Tensor2.default([], [nbRow, nbCol, 1]);
    let offset = 0;
    for (let c = 0; c < inputChannels; c++) {
      for (let i = 0, limit = inputRows - nbRow; i <= limit; i += this.strides[0]) {
        for (let j = 0, limit = inputCols - nbCol; j <= limit; j += this.strides[1]) {
          _ndarrayOps2.default.assign(patch.tensor, x.tensor.hi(i + nbRow, j + nbCol, c + 1).lo(i, j, c));
          this.imColsMat.tensor.data.set(patch.tensor.data, offset);
          offset += patchLen;
        }
      }
    }

    if (this.gpu) {
      this.imColsMat.createGLTexture();
    }
    return this.imColsMat;
  }

  /**
   * @returns {Tensor}
   */
  _w2row() {
    const inputChannels = this.weights['kernel'].tensor.shape[2];
    const [nbFilter, nbRow, nbCol] = this.kernelShape;
    const patchLen = nbRow * nbCol;

    this.wRowsMat = new _Tensor2.default([], [patchLen, nbFilter * inputChannels]);

    let patch = new _Tensor2.default([], [nbRow, nbCol]);
    let patchRaveled = new _Tensor2.default([], [patchLen]);
    let p = 0;
    for (let c = 0; c < inputChannels; c++) {
      for (let n = 0; n < nbFilter; n++) {
        _ndarrayOps2.default.assign(patch.tensor, this.weights['kernel'].tensor.pick(null, null, c, n));
        patchRaveled.replaceTensorData(patch.tensor.data);
        _ndarrayOps2.default.assign(this.wRowsMat.tensor.pick(null, p), patchRaveled.tensor);
        p += 1;
      }
    }

    return this.wRowsMat;
  }

  /**
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
    const inputChannels = this.inputShape[2];
    const matMul = new _Tensor2.default([], [nbPatches * inputChannels, nbFilter * inputChannels]);

    (0, _ndarrayGemm2.default)(matMul.tensor, this.imColsMat.tensor, this.wRowsMat.tensor, 1, 1);

    this.output = new _Tensor2.default([], this.outputShape);

    const outputDataLength = outputRows * outputCols * nbFilter * inputChannels;
    let dataFiltered = new Float32Array(outputDataLength);
    for (let c = 0; c < inputChannels; c++) {
      for (let n = c * outputDataLength + c * nbFilter; n < (c + 1) * outputDataLength; n += nbFilter * inputChannels) {
        for (let m = 0; m < nbFilter; m++) {
          dataFiltered[n + m - c * outputDataLength] = matMul.tensor.data[n + m];
        }
      }
    }
    this.output.replaceTensorData(dataFiltered);
  }

  _createOutputReshapeMap() {
    if (this.reshapeRowIndexMap && this.reshapeColIndexMap) {
      return;
    }

    const nbFilter = this.kernelShape[0];
    const reshape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]];
    this.reshapeRowIndexMap = new _Tensor2.default([], reshape, { type: Int32Array });
    this.reshapeColIndexMap = new _Tensor2.default([], reshape, { type: Int32Array });
    for (let j = 0; j < reshape[1]; j++) {
      for (let i = 0; i < reshape[0]; i++) {
        _ndarrayOps2.default.assigns(this.reshapeRowIndexMap.tensor.pick(i, j), i + Math.floor(j / nbFilter) * reshape[0]);
      }
    }
    for (let j = 0; j < reshape[1]; j++) {
      _ndarrayOps2.default.assigns(this.reshapeColIndexMap.tensor.pick(null, j), j);
    }

    if (this.gpu) {
      this.reshapeRowIndexMap.createGLTexture('2d', 'int');
      this.reshapeColIndexMap.createGLTexture('2d', 'int');
    }
  }

  /**
   * @param {Tensor} x
   */
  _callGPU(x) {
    super._callGPU(x);

    this._createOutputReshapeMap();
    if (!this.outputReshaped) {
      const reshape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]];
      this.outputReshaped = new _Tensor2.default([], reshape);
      this.outputReshaped.createGLTexture();
      this.outputReshaped.is2DReshaped = true;
      this.outputReshaped.originalShape = this.outputShape;
    }

    _WebGL.webgl2.runProgram({
      program: this.mapInputProgram,
      output: this.outputReshaped,
      inputs: [{ texture: this.output.glTexture, type: '2d', name: 'x' }, { texture: this.reshapeRowIndexMap.glTexture, type: '2d', name: 'rowIndexMap' }, { texture: this.reshapeColIndexMap.glTexture, type: '2d', name: 'colIndexMap' }]
    });
  }
}

/**
 * SeparableConv2D layer class
 */
class SeparableConv2D extends _Layer2.default {
  /**
   * Creates a SeparableConv2D layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.filters] - Number of convolution filters to use
   * @param {number|number[]} [attrs.kernel_size] - Size of the convolution kernel
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'SeparableConv2D';

    const {
      filters = 1,
      kernel_size = [1, 1],
      strides = [1, 1],
      padding = 'valid',
      data_format = 'channels_last',
      depth_multiplier = 1,
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

    if (padding === 'valid' || padding === 'same') {
      this.padding = padding;
    } else {
      this.throwError('Invalid padding.');
    }

    this.use_bias = use_bias;

    // Layer weights specification
    this.params = this.use_bias ? ['depthwise_kernel', 'pointwise_kernel', 'bias'] : ['depthwise_kernel', 'pointwise_kernel'];

    // SeparableConv2D has two components: depthwise, and pointwise.
    // Activation function and bias is applied at the end.
    // Subsampling (striding) only performed on depthwise part, not the pointwise part.
    this.depthwiseConvAttrs = {
      filters: depth_multiplier,
      kernel_size: [this.kernelShape[1], this.kernelShape[2]],
      strides: this.strides,
      padding,
      data_format,
      activation: 'linear',
      use_bias: false,
      gpu: attrs.gpu
    };
    this.pointwiseConvAttrs = {
      filters,
      kernel_size: [1, 1],
      strides: [1, 1],
      padding,
      data_format,
      activation: 'linear',
      use_bias,
      gpu: attrs.gpu

      // GPU setup
    };if (this.gpu) {
      this.activationProgram = _WebGL.webgl2.compileProgram(require(`../../activations/${this.activation}.glsl`));
    }
  }

  /**
   * Method for setting layer weights. Override `super` method since weights must be set in component Conv2D layers.
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    this._depthwiseConv = new _DepthwiseConv2D(this.depthwiseConvAttrs);
    this._depthwiseConv.setWeights(weightsArr.slice(0, 1));
    this._pointwiseConv = new _Conv2D2.default(this.pointwiseConvAttrs);
    this._pointwiseConv.setWeights(weightsArr.slice(1, 3));
  }

  /**
   * Method for layer computational logic
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
    this._depthwiseConv._callCPU(x);
    this._pointwiseConv._callCPU(this._depthwiseConv.output);
    this.output = this._pointwiseConv.output;
    this.activationFunc(this.output);
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    // prevent GPU -> CPU data transfer by specifying non-empty outbound nodes array on these internal Conv2D layers
    this._depthwiseConv.outbound = [null];
    this._pointwiseConv.outbound = [null];

    this._depthwiseConv._callGPU(x);
    this._pointwiseConv._callGPU(this._depthwiseConv.outputReshaped);

    // Activation
    if (this.activation === 'linear') {
      this.output = this._pointwiseConv.output;
    } else {
      if (!this.output) {
        this.output = new _Tensor2.default([], this._pointwiseConv.output.glTextureShape);
        this.output.createGLTexture();
        this.output.is2DReshaped = true;
        this.output.originalShape = this._pointwiseConv.output.originalShape;
        this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this._pointwiseConv.output.originalShape, false, -1);
      }
      this.outputPreactiv = this._pointwiseConv.output;
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
exports.default = SeparableConv2D;