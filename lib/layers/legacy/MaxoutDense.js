'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _ndarrayBlasLevel = require('ndarray-blas-level2');

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * MaxoutDense layer class
 * From Keras docs: takes the element-wise maximum of nb_feature Dense(input_dim, output_dim) linear layers
 * Note that `nb_feature` is implicit in the weights tensors, with shapes:
 * - W: [nb_feature, input_dim, output_dim]
 * - b: [nb_feature, output_dim]
 */
class MaxoutDense extends _Layer2.default {
  /**
   * Creates a MaxoutDense layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} attrs.output_dim - output dimension size
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'MaxoutDense';

    const { output_dim = 1, input_dim = null, bias = true } = attrs;
    this.outputDim = output_dim;
    this.inputDim = input_dim;
    this.bias = bias;

    // Layer weights specification
    this.params = this.bias ? ['W', 'b'] : ['W'];
  }

  /**
   * Method for layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    const nbFeature = this.weights['W'].tensor.shape[0];

    let featMax = new _Tensor2.default([], [this.outputDim]);
    for (let i = 0; i < nbFeature; i++) {
      let y = new _Tensor2.default([], [this.outputDim]);
      if (this.bias) {
        _ndarrayOps2.default.assign(y.tensor, this.weights['b'].tensor.pick(i, null));
      }
      (0, _ndarrayBlasLevel.gemv)(1.0, this.weights['W'].tensor.pick(i, null, null).transpose(1, 0), x.tensor, 1.0, y.tensor);
      _ndarrayOps2.default.maxeq(featMax.tensor, y.tensor);
    }

    x.tensor = featMax.tensor;
    return x;
  }
}
exports.default = MaxoutDense;