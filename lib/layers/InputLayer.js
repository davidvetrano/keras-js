'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _isEqual2 = require('lodash/isEqual');

var _isEqual3 = _interopRequireDefault(_isEqual2);

var _Layer = require('../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * InputLayer layer class
 */
class InputLayer extends _Layer2.default {
  /**
   * Creates an InputLayer layer
   *
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'InputLayer';

    const { shape = [] } = attrs;

    this.shape = attrs.batch_input_shape && attrs.batch_input_shape.length ? attrs.batch_input_shape.slice(1) : shape;
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
    this.inputShape = x.tensor.shape;
    if (!(0, _isEqual3.default)(this.inputShape, this.shape)) {
      this.throwError(`input tensor shape ${x.tensor.shape} does not match specified shape ${this.shape}.`);
    }
    this.output = x;
  }

  /**
  * GPU call
  *
  * @param {Tensor} x
  */
  _callGPU(x) {
    if (!x.glTexture) {
      this.inputShape = x.tensor.shape;
    } else {
      if (x.is2DReshaped) {
        this.inputShape = x.originalShape;
      } else {
        this.inputShape = x.tensor.shape;
      }
    }

    if (!(0, _isEqual3.default)(this.inputShape, this.shape)) {
      this.throwError(`input tensor shape ${x.tensor.shape} does not match specified shape ${this.shape}.`);
    }

    if (!x.glTexture) {
      if (x.tensor.shape.length <= 2) {
        x.createGLTexture('2d', 'float');
      } else if (x.tensor.shape.length > 2) {
        x.reshapeTo2DSquare();
        x.createGLTexture('2d', 'float');
      }
    }

    this.output = x;
  }
}
exports.default = InputLayer;