'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * ELU advanced activation layer class
 */
class ELU extends _Layer2.default {
  /**
   * Creates a ELU activation layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.alpha] - scale for the negative factor
   */
  constructor(attrs = {}) {
    super(attrs);

    _initialiseProps.call(this);

    this.layerClass = 'ELU';

    const { alpha = 1.0 } = attrs;

    this.alpha = alpha;

    // GPU setup
    if (this.gpu) {
      this.program = _WebGL.webgl2.compileProgram(require('./ELU.glsl'));
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
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    this.output = x;
    this._compute(this.output.tensor, this.alpha);
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

    if (!this.output) {
      this.output = new _Tensor2.default([], x.glTextureShape);
      this.output.createGLTexture();
      if (x.is1D) {
        this.output.is1D = x.is1D;
      } else if (x.is2DReshaped) {
        this.output.is2DReshaped = x.is2DReshaped;
        this.output.originalShape = x.originalShape;
        this.output.indicesForReshaped = x.indicesForReshaped;
      }
    }

    _WebGL.webgl2.runProgram({
      program: this.program,
      output: this.output,
      inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }],
      uniforms: [{ value: this.alpha, type: 'float', name: 'alpha' }]
    });

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
      if (this.output.is2DReshaped) {
        this.output.reshapeFrom2D();
      }
    }
  }
}
exports.default = ELU;

var _initialiseProps = function () {
  this._compute = (0, _cwise2.default)({
    args: ['array', 'scalar'],
    body: function (_x, alpha) {
      _x = Math.max(_x, 0) + alpha * (Math.exp(Math.min(_x, 0)) - 1);
    }
  });
};