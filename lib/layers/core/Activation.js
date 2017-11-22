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

var activations = _interopRequireWildcard(_activations);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Activation layer class
 */
class Activation extends _Layer2.default {
  /**
   * Creates an Activation layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {String} [attrs.activation] - name of activation function
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Activation';

    const { activation = 'linear' } = attrs;

    this.activation = activation;
    this.activationFunc = activations[activation];

    // GPU setup
    if (this.gpu) {
      this.program = _WebGL.webgl2.compileProgram(require(`../../activations/${this.activation}.glsl`));
    }
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (this.activation === 'linear') {
      this.output = x;
      return this.output;
    }

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
    this.activationFunc(this.output);
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
      inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }]
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
exports.default = Activation;