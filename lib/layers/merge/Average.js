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
 * Average merge layer class, extends abstract _Merge class
 */
class Average extends _Merge3.default {
  /**
   * Creates a Average merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Average';

    this.mode = 'ave';

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = _WebGL.webgl2.compileProgram(require('./Average.glsl'));
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

    for (let i = 0; i < inputs.length; i++) {
      _ndarrayOps2.default.addeq(this.output.tensor, inputs[i].tensor);
    }
    _ndarrayOps2.default.divseq(this.output.tensor, inputs.length);
  }
}
exports.default = Average;