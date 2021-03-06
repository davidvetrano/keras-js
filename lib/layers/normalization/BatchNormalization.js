'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * BatchNormalization layer class
 */
class BatchNormalization extends _Layer2.default {
  /**
   * Creates an BatchNormalization layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'BatchNormalization';

    const { epsilon = 0.001, axis = -1, center = true, scale = true } = attrs;

    this.epsilon = epsilon;
    this.center = center;
    this.scale = scale;

    // no batch axis, so axis is less 1 compared to representation in keras
    // will be set in call(), as input tensor shape is needed to calculate axis
    // if axis < 0
    this.axis = axis;
    this.axisNormalized = false;

    // Layer weights specification
    this.params = [];
    if (this.scale) {
      this.params.push('gamma');
    }
    if (this.center) {
      this.params.push('beta');
    }
    this.params = this.params.concat(['moving_mean', 'moving_variance']);

    // GPU setup
    if (this.gpu) {
      this.program = _WebGL.webgl2.compileProgram(require('./BatchNormalization.glsl'));
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
    if (!this.axisNormalized) {
      this.axis = this.axis < 0 ? x.tensor.shape.length + this.axis : this.axis - 1;
      this.axisNormalized = true;
    }

    let broadcast = [];
    for (let d = 0; d < x.tensor.shape.length; d++) {
      if (d === this.axis) broadcast.push(1);else broadcast.push(null);
    }

    // broadcast weights
    let _gamma = new _Tensor2.default([], x.tensor.shape);
    let _beta = new _Tensor2.default([], x.tensor.shape);
    for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
      broadcast[this.axis] = i;
      if (this.scale) {
        _ndarrayOps2.default.assigns(_gamma.tensor.pick(...broadcast), this.weights['gamma'].tensor.get(i));
      }
      if (this.center) {
        _ndarrayOps2.default.assigns(_beta.tensor.pick(...broadcast), this.weights['beta'].tensor.get(i));
      }
    }

    let _mean = new _Tensor2.default([], x.tensor.shape);
    let _std = new _Tensor2.default([], x.tensor.shape);

    // feature-wise normalization
    for (let i = 0; i < x.tensor.shape[this.axis]; i++) {
      broadcast[this.axis] = i;
      _ndarrayOps2.default.assigns(_mean.tensor.pick(...broadcast), this.weights['moving_mean'].tensor.get(i));
      _ndarrayOps2.default.assigns(_std.tensor.pick(...broadcast), this.weights['moving_variance'].tensor.get(i) + this.epsilon);
    }
    _ndarrayOps2.default.sqrteq(_std.tensor);

    this.output = new _Tensor2.default(x.tensor.data, x.tensor.shape);

    _ndarrayOps2.default.subeq(this.output.tensor, _mean.tensor);
    _ndarrayOps2.default.diveq(this.output.tensor, _std.tensor);
    if (this.scale) {
      _ndarrayOps2.default.muleq(this.output.tensor, _gamma.tensor);
    }
    if (this.center) {
      _ndarrayOps2.default.addeq(this.output.tensor, _beta.tensor);
    }
  }

  /**
   * GPU call
   * (will only work on the 2D-reshaped representation for post-convolutional BN)
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (!this.axisNormalized) {
      if (x.is2DReshaped) {
        this.inputShape = x.originalShape;
      } else {
        this.inputShape = x.tensor.shape;
      }
      this.axis = this.axis < 0 ? this.inputShape.length + this.axis : this.axis - 1;
      this.axisNormalized = true;
    }

    if (!x.glTexture) {
      if (x.tensor.shape.length <= 2) {
        x.createGLTexture();
      } else if (x.tensor.shape.length > 2 && !x.is2DReshaped) {
        x.reshapeTo2D(this.axis);
        x.createGLTexture();
      }
    }

    // create output textures if doesn't already exist
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

    const programInputs = [{ texture: x.glTexture, type: '2d', name: 'X' }];
    if (this.scale) {
      programInputs.push({ texture: this.weights['gamma'].glTexture, type: '2d', name: 'gamma' });
    }
    if (this.center) {
      programInputs.push({ texture: this.weights['beta'].glTexture, type: '2d', name: 'beta' });
    }
    programInputs.push({ texture: this.weights['moving_mean'].glTexture, type: '2d', name: 'mean' });
    programInputs.push({ texture: this.weights['moving_variance'].glTexture, type: '2d', name: 'std' });
    const programUniforms = [{ value: this.epsilon, type: 'float', name: 'epsilon' }, { value: this.output.glTextureShape[0], type: 'int', name: 'rows' }, { value: this.output.glTextureShape[1], type: 'int', name: 'cols' }, { value: +this.scale, type: 'bool', name: 'scale' }, { value: +this.center, type: 'bool', name: 'center' }];
    _WebGL.webgl2.runProgram({
      program: this.program,
      output: this.output,
      inputs: programInputs,
      uniforms: programUniforms
    });

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
      if (this.output.is2DReshaped) {
        this.output.reshapeFrom2D(this.axis);
      }
    }
  }
}
exports.default = BatchNormalization;