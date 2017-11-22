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
 * GRU layer class
 */
class GRU extends _Layer2.default {
  /**
   * Creates a GRU layer
   *
   * @param {Object} [attrs] - layer attributes
   * @param {number} [attrs.units] - output dimensionality
   * @param {number} [attrs.activation] - activation function
   * @param {number} [attrs.recurrent_activation] - inner activation function
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
    this._update = (0, _cwise2.default)({
      args: ['array', 'array', 'array'],
      body: function (_h, _htm1, _z) {
        _h = _h * (1 - _z) + _htm1 * _z;
      }
    });
    this.layerClass = 'GRU';

    const {
      units = 1,
      activation = 'tanh',
      use_bias = true,
      recurrent_activation = 'hard_sigmoid',
      return_sequences = false,
      go_backwards = false,
      stateful = false
    } = attrs;

    this.units = units;

    // keep this.activation and this.recurrentActivation for Bidirectional wrapper layer to use
    this.activation = activation;
    this.recurrentActivation = recurrent_activation;
    this.activationFunc = activations[activation];
    this.recurrentActivationFunc = activations[recurrent_activation];

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
      this.recurrentActivationProgram = _WebGL.webgl2.compileProgram(require(`../../activations/${this.recurrentActivation}.glsl`));
      this.gateSummationProgram = _WebGL.webgl2.compileProgram(require('./gateSummation.glsl'));
      this.gateProductProgram = _WebGL.webgl2.compileProgram(require('./gateProduct.glsl'));
      this.timestepReadProgram = _WebGL.webgl2.compileProgram(require('./timestepRead.glsl'));
      this.timestepWriteProgram = _WebGL.webgl2.compileProgram(require('./timestepWrite.glsl'));
      this.updateProgram = _WebGL.webgl2.compileProgram(require('./GRU.update.glsl'));
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   *
   * W weight tensor is split into W_z, W_r, W_h
   *
   * U weight tensor is split into U_z, U_r, U_h
   *
   * b weight tensor is split into b_z, b_r, b_h (or create empty bias if this.use_bias is false)
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    super.setWeights(weightsArr);

    const shape_W = this.weights['kernel'].tensor.shape;
    this.weights['W_z'] = new _Tensor2.default([], [shape_W[0], this.units]);
    this.weights['W_r'] = new _Tensor2.default([], [shape_W[0], this.units]);
    this.weights['W_h'] = new _Tensor2.default([], [shape_W[0], this.units]);
    _ndarrayOps2.default.assign(this.weights['W_z'].tensor, this.weights['kernel'].tensor.hi(shape_W[0], this.units).lo(0, 0));
    _ndarrayOps2.default.assign(this.weights['W_r'].tensor, this.weights['kernel'].tensor.hi(shape_W[0], 2 * this.units).lo(0, this.units));
    _ndarrayOps2.default.assign(this.weights['W_h'].tensor, this.weights['kernel'].tensor.hi(shape_W[0], 3 * this.units).lo(0, 2 * this.units));

    const shape_U = this.weights['recurrent_kernel'].tensor.shape;
    this.weights['U_z'] = new _Tensor2.default([], [shape_U[0], this.units]);
    this.weights['U_r'] = new _Tensor2.default([], [shape_U[0], this.units]);
    this.weights['U_h'] = new _Tensor2.default([], [shape_U[0], this.units]);
    _ndarrayOps2.default.assign(this.weights['U_z'].tensor, this.weights['recurrent_kernel'].tensor.hi(shape_U[0], this.units).lo(0, 0));
    _ndarrayOps2.default.assign(this.weights['U_r'].tensor, this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 2 * this.units).lo(0, this.units));
    _ndarrayOps2.default.assign(this.weights['U_h'].tensor, this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 3 * this.units).lo(0, 2 * this.units));

    this.weights['b_z'] = new _Tensor2.default([], [this.units]);
    this.weights['b_r'] = new _Tensor2.default([], [this.units]);
    this.weights['b_h'] = new _Tensor2.default([], [this.units]);
    if (this.use_bias) {
      _ndarrayOps2.default.assign(this.weights['b_z'].tensor, this.weights['bias'].tensor.hi(this.units).lo(0));
      _ndarrayOps2.default.assign(this.weights['b_r'].tensor, this.weights['bias'].tensor.hi(2 * this.units).lo(this.units));
      _ndarrayOps2.default.assign(this.weights['b_h'].tensor, this.weights['bias'].tensor.hi(3 * this.units).lo(2 * this.units));
    }

    if (this.gpu) {
      const names = ['W_z', 'W_r', 'W_h', 'U_z', 'U_r', 'U_h', 'b_z', 'b_r', 'b_h'];
      names.forEach(name => {
        this.weights[name].createGLTexture();
      });
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
    const dimUpdateGate = this.weights['b_z'].tensor.shape[0];
    const dimResetGate = this.weights['b_r'].tensor.shape[0];
    const dimHiddenState = this.weights['b_h'].tensor.shape[0];

    const currentUpdateGateState = new _Tensor2.default([], [dimUpdateGate]);
    const tempXZ = new _Tensor2.default([], [dimUpdateGate]);
    const tempHZ = new _Tensor2.default([], [dimUpdateGate]);

    const currentResetGateState = new _Tensor2.default([], [dimResetGate]);
    const tempXR = new _Tensor2.default([], [dimResetGate]);
    const tempHR = new _Tensor2.default([], [dimResetGate]);

    const currentHiddenState = this.stateful && this.currentHiddenState ? this.currentHiddenState : new _Tensor2.default([], [dimHiddenState]);
    const tempXH = new _Tensor2.default([], [dimHiddenState]);
    const tempHH = new _Tensor2.default([], [dimHiddenState]);
    const previousHiddenState = new _Tensor2.default([], [dimHiddenState]);

    this.hiddenStateSequence = new _Tensor2.default([], [x.tensor.shape[0], dimHiddenState]);

    const currentX = new _Tensor2.default([], [x.tensor.shape[1]]);

    const _step = () => {
      _ndarrayOps2.default.assign(previousHiddenState.tensor, currentHiddenState.tensor);

      (0, _ndarrayBlasLevel.gemv)(1, this.weights['W_z'].tensor.transpose(1, 0), currentX.tensor, 1, tempXZ.tensor);
      (0, _ndarrayBlasLevel.gemv)(1, this.weights['U_z'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHZ.tensor);
      this._combine(currentUpdateGateState.tensor, tempXZ.tensor, tempHZ.tensor, this.weights['b_z'].tensor);
      this.recurrentActivationFunc(currentUpdateGateState);

      (0, _ndarrayBlasLevel.gemv)(1, this.weights['W_r'].tensor.transpose(1, 0), currentX.tensor, 1, tempXR.tensor);
      (0, _ndarrayBlasLevel.gemv)(1, this.weights['U_r'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHR.tensor);
      this._combine(currentResetGateState.tensor, tempXR.tensor, tempHR.tensor, this.weights['b_r'].tensor);
      this.recurrentActivationFunc(currentResetGateState);

      _ndarrayOps2.default.muleq(currentResetGateState.tensor, previousHiddenState.tensor);
      (0, _ndarrayBlasLevel.gemv)(1, this.weights['W_h'].tensor.transpose(1, 0), currentX.tensor, 1, tempXH.tensor);
      (0, _ndarrayBlasLevel.gemv)(1, this.weights['U_h'].tensor.transpose(1, 0), currentResetGateState.tensor, 1, tempHH.tensor);
      this._combine(currentHiddenState.tensor, tempXH.tensor, tempHH.tensor, this.weights['b_h'].tensor);
      this.activationFunc(currentHiddenState);

      this._update(currentHiddenState.tensor, previousHiddenState.tensor, currentUpdateGateState.tensor);
    };

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i;
      _ndarrayOps2.default.assign(currentX.tensor, x.tensor.pick(inputIndex, null));

      // clear temp tensors
      const tempTensors = [tempXZ, tempHZ, tempXR, tempHR, tempXH, tempHH];
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

    // update gate

    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXZ,
      inputs: [{ texture: this.currentX.glTexture, type: '2d', name: 'A' }, { texture: this.weights['W_z'].glTexture, type: '2d', name: 'B' }],
      uniforms: [{ value: 0, type: 'bool', name: 'addC' }, { value: 1, type: 'int', name: 'M' }, { value: this.weights['W_z'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['W_z'].glTextureShape[1], type: 'int', name: 'N' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHZ,
      inputs: [{ texture: this.previousHiddenState.glTexture, type: '2d', name: 'A' }, { texture: this.weights['U_z'].glTexture, type: '2d', name: 'B' }],
      uniforms: [{ value: 0, type: 'bool', name: 'addC' }, { value: 1, type: 'int', name: 'M' }, { value: this.weights['U_z'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['U_z'].glTextureShape[1], type: 'int', name: 'N' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentUpdateGateStatePreactiv,
      inputs: [{ texture: this.tempXZ.glTexture, type: '2d', name: 't1' }, { texture: this.tempHZ.glTexture, type: '2d', name: 't2' }, { texture: this.weights['b_z'].glTexture, type: '2d', name: 'bias' }]
    });

    if (this.recurrentActivation !== 'linear') {
      _WebGL.webgl2.runProgram({
        program: this.recurrentActivationProgram,
        output: this.currentUpdateGateState,
        inputs: [{ texture: this.currentUpdateGateStatePreactiv.glTexture, type: '2d', name: 'x' }]
      });
    } else {
      this.currentUpdateGateState = this.currentUpdateGateStatePreactiv;
    }

    // reset gate

    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXR,
      inputs: [{ texture: this.currentX.glTexture, type: '2d', name: 'A' }, { texture: this.weights['W_r'].glTexture, type: '2d', name: 'B' }],
      uniforms: [{ value: 0, type: 'bool', name: 'addC' }, { value: 1, type: 'int', name: 'M' }, { value: this.weights['W_r'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['W_r'].glTextureShape[1], type: 'int', name: 'N' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHR,
      inputs: [{ texture: this.previousHiddenState.glTexture, type: '2d', name: 'A' }, { texture: this.weights['U_r'].glTexture, type: '2d', name: 'B' }],
      uniforms: [{ value: 0, type: 'bool', name: 'addC' }, { value: 1, type: 'int', name: 'M' }, { value: this.weights['U_r'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['U_r'].glTextureShape[1], type: 'int', name: 'N' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentResetGateStatePreactiv,
      inputs: [{ texture: this.tempXR.glTexture, type: '2d', name: 't1' }, { texture: this.tempHR.glTexture, type: '2d', name: 't2' }, { texture: this.weights['b_r'].glTexture, type: '2d', name: 'bias' }]
    });

    if (this.recurrentActivation !== 'linear') {
      _WebGL.webgl2.runProgram({
        program: this.recurrentActivationProgram,
        output: this.currentResetGateState,
        inputs: [{ texture: this.currentResetGateStatePreactiv.glTexture, type: '2d', name: 'x' }]
      });
    } else {
      this.currentResetGateState = this.currentResetGateStatePreactiv;
    }

    // hidden state

    _WebGL.webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.currentResetGateStateCopy,
      inputs: [{ texture: this.currentResetGateState.glTexture, type: '2d', name: 'source' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.gateProductProgram,
      output: this.currentResetGateState,
      inputs: [{ texture: this.currentResetGateStateCopy.glTexture, type: '2d', name: 't1' }, { texture: this.previousHiddenState.glTexture, type: '2d', name: 't2' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXH,
      inputs: [{ texture: this.currentX.glTexture, type: '2d', name: 'A' }, { texture: this.weights['W_h'].glTexture, type: '2d', name: 'B' }],
      uniforms: [{ value: 0, type: 'bool', name: 'addC' }, { value: 1, type: 'int', name: 'M' }, { value: this.weights['W_h'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['W_h'].glTextureShape[1], type: 'int', name: 'N' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHH,
      inputs: [{ texture: this.currentResetGateState.glTexture, type: '2d', name: 'A' }, { texture: this.weights['U_h'].glTexture, type: '2d', name: 'B' }],
      uniforms: [{ value: 0, type: 'bool', name: 'addC' }, { value: 1, type: 'int', name: 'M' }, { value: this.weights['U_h'].glTextureShape[0], type: 'int', name: 'K' }, { value: this.weights['U_h'].glTextureShape[1], type: 'int', name: 'N' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentHiddenStatePreactiv,
      inputs: [{ texture: this.tempXH.glTexture, type: '2d', name: 't1' }, { texture: this.tempHH.glTexture, type: '2d', name: 't2' }, { texture: this.weights['b_h'].glTexture, type: '2d', name: 'bias' }]
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

    _WebGL.webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.currentHiddenStateCopy,
      inputs: [{ texture: this.currentHiddenState.glTexture, type: '2d', name: 'source' }]
    });

    _WebGL.webgl2.runProgram({
      program: this.updateProgram,
      output: this.currentHiddenState,
      inputs: [{ texture: this.currentHiddenStateCopy.glTexture, type: '2d', name: 'h' }, { texture: this.previousHiddenState.glTexture, type: '2d', name: 'htm1' }, { texture: this.currentUpdateGateState.glTexture, type: '2d', name: 'z' }]
    });
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

    const dimUpdateGate = this.weights['b_z'].glTextureShape[1];
    const dimResetGate = this.weights['b_r'].glTextureShape[1];
    const dimHiddenState = this.weights['b_h'].glTextureShape[1];

    if (!this.currentHiddenState || !this.stateful) {
      this.currentHiddenState = new _Tensor2.default([], [dimHiddenState]);
      this.currentHiddenState.createGLTexture();
    }
    if (!this.currentHiddenStateCopy) {
      this.currentHiddenStateCopy = new _Tensor2.default([], [dimHiddenState]);
      this.currentHiddenStateCopy.createGLTexture();
    }
    if (!this.currentHiddenStatePreactiv) {
      this.currentHiddenStatePreactiv = new _Tensor2.default([], [dimHiddenState]);
      this.currentHiddenStatePreactiv.createGLTexture();
    }

    if (!this.currentUpdateGateState) {
      this.currentUpdateGateState = new _Tensor2.default([], [dimUpdateGate]);
      this.currentUpdateGateState.createGLTexture();
    }
    if (!this.currentUpdateGateStatePreactiv) {
      this.currentUpdateGateStatePreactiv = new _Tensor2.default([], [dimUpdateGate]);
      this.currentUpdateGateStatePreactiv.createGLTexture();
    }
    if (!this.tempXZ) {
      this.tempXZ = new _Tensor2.default([], [dimUpdateGate]);
      this.tempXZ.createGLTexture();
    }
    if (!this.tempHZ) {
      this.tempHZ = new _Tensor2.default([], [dimUpdateGate]);
      this.tempHZ.createGLTexture();
    }

    if (!this.currentResetGateState) {
      this.currentResetGateState = new _Tensor2.default([], [dimResetGate]);
      this.currentResetGateState.createGLTexture();
    }
    if (!this.currentResetGateStateCopy) {
      this.currentResetGateStateCopy = new _Tensor2.default([], [dimResetGate]);
      this.currentResetGateStateCopy.createGLTexture();
    }
    if (!this.currentResetGateStatePreactiv) {
      this.currentResetGateStatePreactiv = new _Tensor2.default([], [dimResetGate]);
      this.currentResetGateStatePreactiv.createGLTexture();
    }
    if (!this.tempXR) {
      this.tempXR = new _Tensor2.default([], [dimResetGate]);
      this.tempXR.createGLTexture();
    }
    if (!this.tempHR) {
      this.tempHR = new _Tensor2.default([], [dimResetGate]);
      this.tempHR.createGLTexture();
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
exports.default = GRU;