'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Merge2 = require('./_Merge');

var _Merge3 = _interopRequireDefault(_Merge2);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _ndarrayGemm = require('ndarray-gemm');

var _ndarrayGemm2 = _interopRequireDefault(_ndarrayGemm);

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Dot merge layer class, extends abstract _Merge class
 */
class Dot extends _Merge3.default {
  /**
   * Creates a Dot merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Dot';

    this.mode = 'dot';

    const { axes = -1, normalize = false } = attrs;

    // no mini-batch axis here, so we subtract 1 if given axis > 0
    if (Array.isArray(axes)) {
      this.dotAxes = [axes[0] <= 0 ? axes[0] : axes[0] - 1, axes[1] <= 0 ? axes[1] : axes[1] - 1];
    } else {
      this.dotAxes = [axes <= 0 ? axes : axes - 1, axes <= 0 ? axes : axes - 1];
    }

    this.normalize = normalize;

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = _WebGL.webgl2.compileProgram(require('./Dot.glsl'));
    }
  }

  /**
   * Calculate output shape
   *
   * @param {number[][]} inputShapes
   */
  _calcOutputShape(inputShapes) {
    let shape1 = inputShapes[0].slice();
    let shape2 = inputShapes[1].slice();
    shape1.splice(this.dotAxes[0], 1);
    shape2.splice(this.dotAxes[1], 1);
    this.outputShape = shape1.concat(shape2);
    if (this.outputShape.length === 1) {
      this.outputShape.push(1);
    }
  }

  /**
   * CPU call
   *
   * @param {Tensor[]} inputs
   */
  _callCPU(inputs) {
    this._calcOutputShape([inputs[0].tensor.shape, inputs[1].tensor.shape]);
    this.output = new _Tensor2.default([], this.outputShape);

    if (inputs[0].tensor.shape.length === 2 && inputs[1].tensor.shape.length === 2) {
      if (this.dotAxes[0] === 0 && this.dotAxes[1] === 0) {
        if (this.normalize) {
          for (let i = 0; i < inputs[0].tensor.shape[1]; i++) {
            _ndarrayOps2.default.divseq(inputs[0].tensor.pick(null, i), _ndarrayOps2.default.norm2(inputs[0].tensor.pick(null, i)));
          }
          for (let i = 0; i < inputs[1].tensor.shape[1]; i++) {
            _ndarrayOps2.default.divseq(inputs[1].tensor.pick(null, i), _ndarrayOps2.default.norm2(inputs[1].tensor.pick(null, i)));
          }
        }
        (0, _ndarrayGemm2.default)(this.output.tensor, inputs[0].tensor.transpose(1, 0), inputs[1].tensor);
      } else if (this.dotAxes[0] === 1 && this.dotAxes[1] === 1) {
        if (this.normalize) {
          for (let i = 0; i < inputs[0].tensor.shape[0]; i++) {
            _ndarrayOps2.default.divseq(inputs[0].tensor.pick(i, null), _ndarrayOps2.default.norm2(inputs[0].tensor.pick(i, null)));
          }
          for (let i = 0; i < inputs[1].tensor.shape[0]; i++) {
            _ndarrayOps2.default.divseq(inputs[1].tensor.pick(i, null), _ndarrayOps2.default.norm2(inputs[1].tensor.pick(i, null)));
          }
        }
        (0, _ndarrayGemm2.default)(this.output.tensor, inputs[0].tensor, inputs[1].tensor.transpose(1, 0));
      }
    } else {
      this.throwError('dot mode for 3+ dim tensors not yet implemented.');
    }
  }

  /**
   * GPU call
   *
   * @param {Tensor[]} inputs
   */
  _callGPU(inputs) {
    this._calcOutputShape([inputs[0].glTextureShape, inputs[1].glTextureShape]);

    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new _Tensor2.default([], this.outputShape);
      this.output.createGLTexture();
    }

    const commonDim = inputs[0].glTextureShape[this.dotAxes[0]];

    _WebGL.webgl2.runProgram({
      program: this.mergeProgram,
      output: this.output,
      inputs: [{ texture: inputs[0].glTexture, type: '2d', name: 'input1' }, { texture: inputs[1].glTexture, type: '2d', name: 'input2' }],
      uniforms: [{ value: this.output.glTextureShape[0], type: 'int', name: 'rows' }, { value: this.output.glTextureShape[1], type: 'int', name: 'cols' }, { value: this.dotAxes[0], type: 'int', name: 'dotAxis1' }, { value: this.dotAxes[1], type: 'int', name: 'dotAxis2' }, { value: commonDim, type: 'int', name: 'commonDim' }, { value: +this.normalize, type: 'bool', name: 'normalize' }]
    });

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
    }
  }
}
exports.default = Dot;