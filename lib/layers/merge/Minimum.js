'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Merge2 = require('./_Merge');

var _Merge3 = _interopRequireDefault(_Merge2);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Minimum merge layer class, extends abstract _Merge class
 */
class Minimum extends _Merge3.default {
  /**
   * Creates a Minimum merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Minimum';

    this.mode = 'min';

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = _WebGL.webgl2.compileProgram(require('./Minimum.glsl'));
    }
  }

  /**
   * CPU call
   *
   * @param {Tensor[]} inputs
   */
  _callCPU(inputs) {
    const outputShape = inputs[0].tensor.shape.slice();
    this.output = new _Tensor2.default([], outputShape);

    _ndarrayOps2.default.assign(this.output.tensor, inputs[0].tensor);
    for (let i = 1; i < inputs.length; i++) {
      _ndarrayOps2.default.mineq(this.output.tensor, inputs[i].tensor);
    }
  }
}
exports.default = Minimum;