'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _activations = require('../../activations');

var activations = _interopRequireWildcard(_activations);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _ndarrayBlasLevel = require('ndarray-blas-level2');

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

/**
 * Highway layer class
 * From Keras docs: Densely connected highway network, a natural extension of LSTMs to feedforward networks.
 */
class Highway extends _Layer2.default {
  /**
   * Creates a Highway layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this._computeOutput = (0, _cwise2.default)({
      args: ['array', 'array', 'array'],
      body: function (_x, _y, _transform) {
        _x = _y * _transform + (1 - _transform) * _x;
      }
    });
    this.layerClass = 'Highway';

    const { activation = 'linear', bias = true } = attrs;

    this.activation = activation;
    this.activationFunc = activations[activation];

    this.bias = bias;

    /**
     * Layer weights specification
     */
    this.params = this.bias ? ['W', 'W_carry', 'b', 'b_carry'] : ['W', 'W_carry'];
  }

  /**
   * Method for layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    let y = new _Tensor2.default([], [this.weights['W'].tensor.shape[1]]);
    if (this.bias) {
      _ndarrayOps2.default.assign(y.tensor, this.weights['b'].tensor);
    }
    (0, _ndarrayBlasLevel.gemv)(1.0, this.weights['W'].tensor.transpose(1, 0), x.tensor, 1.0, y.tensor);
    this.activationFunc(y);

    let transform = new _Tensor2.default([], [this.weights['W_carry'].tensor.shape[1]]);
    if (this.bias) {
      _ndarrayOps2.default.assign(transform.tensor, this.weights['b_carry'].tensor);
    }
    (0, _ndarrayBlasLevel.gemv)(1.0, this.weights['W_carry'].tensor.transpose(1, 0), x.tensor, 1.0, transform.tensor);
    activations.sigmoid(transform);

    this._computeOutput(x.tensor, y.tensor, transform.tensor);

    return x;
  }
}
exports.default = Highway;