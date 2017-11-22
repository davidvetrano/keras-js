'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _activations = require('../../activations');

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * LeakyReLU advanced activation layer class
 */
class LeakyReLU extends _Layer2.default {
  /**
   * Creates a LeakyReLU activation layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.alpha] - negative slope coefficient
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'LeakyReLU';

    const { alpha = 0.3 } = attrs;

    this.alpha = alpha;

    // GPU setup
    if (this.gpu) {
      this.program = _WebGL.webgl2.compileProgram(require('./LeakyReLU.glsl'));
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
    (0, _activations.relu)(this.output, { alpha: this.alpha });
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
exports.default = LeakyReLU;