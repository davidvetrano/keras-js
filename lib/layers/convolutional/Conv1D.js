'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _Conv2D = require('./Conv2D');

var _Conv2D2 = _interopRequireDefault(_Conv2D);

var _tensorUtils = require('../../utils/tensorUtils');

var tensorUtils = _interopRequireWildcard(_tensorUtils);

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

var _ndarraySqueeze = require('ndarray-squeeze');

var _ndarraySqueeze2 = _interopRequireDefault(_ndarraySqueeze);

var _ndarrayUnsqueeze = require('ndarray-unsqueeze');

var _ndarrayUnsqueeze2 = _interopRequireDefault(_ndarrayUnsqueeze);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Conv1D layer class
 */
class Conv1D extends _Layer2.default {
  /**
   * Creates a Conv1D layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.filters] - Number of convolution filters to use
   * @param {number} [attrs.kernel_size] - Length of 1D convolution kernel
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Conv1D';

    const {
      filters = 1,
      kernel_size = 1,
      strides = 1,
      padding = 'valid',
      dilation_rate = 1,
      activation = 'linear',
      use_bias = true
    } = attrs;

    if (padding !== 'valid' && padding !== 'same') {
      this.throwError('Invalid padding.');
    }

    if (dilation_rate !== 1 && strides !== 1) {
      // Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1
      // https://keras.io/layers/convolutional/#conv1d
      this.throwError('Incompatible combination of dilation_rate with strides.');
    }

    this.use_bias = use_bias;

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'bias'] : ['kernel'];

    // Bootstrap Conv2D layer:
    // Conv1D is actually a shim on top of Conv2D, where
    // all of the computational action is performed
    // Note that we use `channels_first` dim ordering here.
    const conv2dAttrs = {
      filters,
      kernel_size: [kernel_size, 1],
      strides: [strides, 1],
      padding,
      data_format: 'channels_first',
      dilation_rate,
      activation,
      use_bias
    };
    this._conv2dAttrs = conv2dAttrs;
    this._conv2d = new _Conv2D2.default(Object.assign(conv2dAttrs, { gpu: attrs.gpu }));
  }

  /**
   * Method for setting layer weights
   *
   * Override `super` method since weights must be set in `this._conv2d`
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    weightsArr[0].tensor = (0, _ndarrayUnsqueeze2.default)(weightsArr[0].tensor).transpose(2, 1, 0, 3);
    this._conv2d.setWeights(weightsArr);
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
    const input = new _Tensor2.default(x.tensor.data, x.tensor.shape);
    input.tensor = (0, _ndarrayUnsqueeze2.default)(input.tensor).transpose(0, 2, 1);
    const conv2dOutput = this._conv2d.call(input);
    this.outputShape = [0, 2].map(i => this._conv2d.outputShape[i]);
    this.output = new _Tensor2.default([], this.outputShape);
    _ndarrayOps2.default.assign(this.output.tensor, (0, _ndarraySqueeze2.default)(conv2dOutput.tensor).transpose(1, 0, 2));
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
    const inputShape = x.tensor.shape;
    const input = new _Tensor2.default([], inputShape);
    input.glTexture = x.glTexture;
    input.glTextureShape = inputShape;
    input.is2DReshaped = true;
    input.originalShape = [inputShape[0], 1, inputShape[1]];
    input.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(input.originalShape, false, -1);

    this.output = this._conv2d.call(input);

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
    }
  }
}
exports.default = Conv1D;