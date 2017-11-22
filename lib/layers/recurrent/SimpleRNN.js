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

var _WebGL = require('../../WebGL2');

var _ndarrayBlasLevel = require('ndarray-blas-level2');

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

/**
 * SimpleRNN layer class
 */
class SimpleRNN extends _Layer2.default {
  /**
   * Creates a SimpleRNN layer
   *
   * @param {Object} [attrs] - layer attributes
   * @param {number} [attrs.units] - output dimensionality
   * @param {number} [attrs.activation] - activation function
   * @param {number} [attrs.use_bias] - use bias
   * @param {number} [attrs.return_sequences] - return the last output in the output sequence or the full sequence
   * @param {number} [attrs.go_backwards] - process the input sequence backwards
   * @param {number} [attrs.stateful] - whether to save the last state as the initial state for the next pass
   */
  constructor(attrs = {}) {
    super(attrs);
    this._combine = (0, _cwise2.default)({
      args: ['array', 'array', 'array', 'array'],
      body: function (_y, _x1, _x2, _b) {
        _y = _x1 + _x2 + _b;
      }
    });
    this.layerClass = 'SimpleRNN';

    const {
      units = 1,
      activation = 'tanh',
      use_bias = true,
      return_sequences = false,
      go_backwards = false,
      stateful = false
    } = attrs;

    this.units = units;

    // keep this.activation for Bidirectional wrapper layer to use
    this.activation = activation;
    this.activationFunc = activations[activation];

    this.use_bias = use_bias;

    this.returnSequences = return_sequences;
    this.goBackwards = go_backwards;
    this.stateful = stateful;

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'recurrent_kernel', 'bias'] : ['kernel', 'recurrent_kernel'];

    // GPU setup
    if (this.gpu) {
      this.copyTextureProgram = _WebGL.webgl2.compileProgram(require('../../copyTexture.glsl'));
      this.matMulProgram = _WebGL.webgl2.compileProgram(require('../../matMul.glsl'));
      this.activationProgram = _WebGL.webgl2.compileProgram(require(`../../activations/${this.activation}.glsl`));
      this.gateSummationProgram = _WebGL.webgl2.compileProgram(require('./gateSummation.glsl'));
      this.timestepReadProgram = _WebGL.webgl2.compileProgram(require('./timestepRead.glsl'));
      this.timestepWriteProgram = _WebGL.webgl2.compileProgram(require('./timestepWrite.glsl'));
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method. Create empty bias if this.use_bias is false.
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    super.setWeights(weightsArr);
    if (!this.use_bias) {
      this.weights['bias'] = new _Tensor2.default([], [this.units]);
      if (this.gpu) {
        this.weights['bias'].createGLTexture();
      }
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
    const dimHiddenState = this.units;

    const currentHiddenState = this.stateful && this.currentHiddenState ? this.currentHiddenState : new _Tensor2.default([], [dimHiddenState]);
    const tempXH = new _Tensor2.default([], [dimHiddenState]);
    const tempHH = new _Tensor2.default([], [dimHiddenState]);
    const previousHiddenState = new _Tensor2.default([], [dimHiddenState]);

    this.hiddenStateSequence = new _Tensor2.default([], [x.tensor.shape[0], dimHiddenState]);

    const currentX = new _Tensor2.default([], [x.tensor.shape[1]]);

    const _step = () => {
      _ndarrayOps2.default.assign(previousHiddenState.tensor, currentHiddenState.tensor);

      (0, _ndarrayBlasLevel.gemv)(1, this.weights['kernel'].tensor.transpose(1, 0), currentX.tensor, 1, tempXH.tensor);
      (0, _ndarrayBlasLevel.gemv)(1, this.weights['recurrent_kernel'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHH.tensor);
      this._combine(currentHiddenState.tensor, tempXH.tensor, tempHH.tensor, this.weights['bias'].tensor);
      this.activationFunc(currentHiddenState);
    };

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i;
      _ndarrayOps2.default.assign(currentX.tensor, x.tensor.pick(inputIndex, null));

      // clear temp tensors
      const tempTensors = [tempXH, tempHH];
      tempTensors.forEach(temp => _ndarrayOps2.default.assigns(temp.tensor, 0));

      // advance timestep
      _step();

      if (this.returnSequences) {
        _ndarrayOps2.default.assign(this.hiddenStateSequence.tensor.pick(i, null), currentHiddenState.tensor);
      }
    }

    if (this.returnSequences) {
      this.output = this.hiddenStateSequence;
    } else {
      this.output = currentHiddenState;
    }

    if (this.stateful) {
      this.currentHiddenState = currentHiddenState;
    }
  }

  /**
   * Advance time step in _callGPU
   */
  _stepGPU() {
    _WebGL.webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.previousHiddenState,
      inputs: [{ texture: this.currentHiddenState.glTexture, type: '2d', name: 'source' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXH,
      inputs: [{ texture: this.currentX.glTexture, type: '2d', name: 'A' }, { texture: this.weights['kernel'].glTexture, type: '2d', name: 'B' }],
      uniforms: [{ value: 0, type: 'bool', name: 'addC' }, { value: 1, type: 'int', name: 'M' }, { value: this.weights['kernel'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['kernel'].glTextureShape[1], type: 'int', name: 'N' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHH,
      inputs: [{ texture: this.previousHiddenState.glTexture, type: '2d', name: 'A' }, { texture: this.weights['recurrent_kernel'].glTexture, type: '2d', name: 'B' }],
      uniforms: [{ value: 0, type: 'bool', name: 'addC' }, { value: 1, type: 'int', name: 'M' }, { value: this.weights['recurrent_kernel'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['recurrent_kernel'].glTextureShape[1], type: 'int', name: 'N' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentHiddenStatePreactiv,
      inputs: [{ texture: this.tempXH.glTexture, type: '2d', name: 't1' }, { texture: this.tempHH.glTexture, type: '2d', name: 't2' }, { texture: this.weights['bias'].glTexture, type: '2d', name: 'bias' }]
    });

    if (this.activation !== 'linear') {
      _WebGL.webgl2.runProgram({
        program: this.activationProgram,
        output: this.currentHiddenState,
        inputs: [{ texture: this.currentHiddenStatePreactiv.glTexture, type: '2d', name: 'x' }]
      });
    } else {
      this.currentHiddenState = this.currentHiddenStatePreactiv;
    }
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

    const dimHiddenState = this.units;

    if (!this.currentHiddenState || !this.stateful) {
      this.currentHiddenState = new _Tensor2.default([], [dimHiddenState]);
      this.currentHiddenState.createGLTexture();
    }
    if (!this.currentHiddenStatePreactiv) {
      this.currentHiddenStatePreactiv = new _Tensor2.default([], [dimHiddenState]);
      this.currentHiddenStatePreactiv.createGLTexture();
    }

    if (!this.tempXH) {
      this.tempXH = new _Tensor2.default([], [dimHiddenState]);
      this.tempXH.createGLTexture();
    }
    if (!this.tempHH) {
      this.tempHH = new _Tensor2.default([], [dimHiddenState]);
      this.tempHH.createGLTexture();
    }

    if (!this.previousHiddenState) {
      this.previousHiddenState = new _Tensor2.default([], [dimHiddenState]);
      this.previousHiddenState.createGLTexture();
    }

    if (!this.hiddenStateSequence) {
      this.hiddenStateSequence = new _Tensor2.default([], [x.glTextureShape[0], dimHiddenState]);
      this.hiddenStateSequence.createGLTexture();
    }
    if (!this.hiddenStateSequenceCopy) {
      this.hiddenStateSequenceCopy = new _Tensor2.default([], [x.glTextureShape[0], dimHiddenState]);
      this.hiddenStateSequenceCopy.createGLTexture();
    }

    if (!this.currentX) {
      this.currentX = new _Tensor2.default([], [x.glTextureShape[1]]);
      this.currentX.createGLTexture();
    }

    for (let i = 0, len = x.glTextureShape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i;

      _WebGL.webgl2.runProgram({
        program: this.timestepReadProgram,
        output: this.currentX,
        inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }],
        uniforms: [{ value: inputIndex, type: 'int', name: 'index' }]
      });

      this._stepGPU();

      if (this.returnSequences) {
        _WebGL.webgl2.runProgram({
          program: this.copyTextureProgram,
          output: this.hiddenStateSequenceCopy,
          inputs: [{ texture: this.hiddenStateSequence.glTexture, type: '2d', name: 'source' }]
        });
        _WebGL.webgl2.runProgram({
          program: this.timestepWriteProgram,
          output: this.hiddenStateSequence,
          inputs: [{ texture: this.currentHiddenState.glTexture, type: '2d', name: 'x' }, { texture: this.hiddenStateSequenceCopy.glTexture, type: '2d', name: 'y' }],
          uniforms: [{ value: i, type: 'int', name: 'index' }]
        });
      }
    }

    if (this.returnSequences) {
      this.output = this.hiddenStateSequence;
    } else {
      this.output = this.currentHiddenState;
    }

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
    }
  }
}
exports.default = SimpleRNN;