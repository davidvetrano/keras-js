import * as activations from '../../activations'
import Tensor from '../../Tensor'
import Layer from '../../Layer'
import { webgl2 } from '../../WebGL2'
import { gemv } from 'ndarray-blas-level2'
import ops from 'ndarray-ops'
import cwise from 'cwise'

/**
 * LSTM layer class
 */
export default class LSTM extends Layer {
  /**
   * Creates a LSTM layer
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
    super(attrs)
    this.layerClass = 'LSTM'

    const {
      units = 1,
      activation = 'tanh',
      use_bias = true,
      recurrent_activation = 'hard_sigmoid',
      return_sequences = false,
      go_backwards = false,
      stateful = false
    } = attrs

    this.units = units

    // keep this.activation and this.recurrentActivation for Bidirectional wrapper layer to use
    this.activation = activation
    this.recurrentActivation = recurrent_activation
    this.activationFunc = activations[activation]
    this.recurrentActivationFunc = activations[recurrent_activation]

    this.use_bias = use_bias

    this.returnSequences = return_sequences
    this.goBackwards = go_backwards
    this.stateful = stateful

    // Layer weights specification
    this.params = this.use_bias ? ['kernel', 'recurrent_kernel', 'bias'] : ['kernel', 'recurrent_kernel']

    // GPU setup
    if (this.gpu) {
      this.copyTextureProgram = webgl2.compileProgram(require('../../copyTexture.glsl'))
      this.matMulProgram = webgl2.compileProgram(require('../../matMul.glsl'))
      this.activationProgram = webgl2.compileProgram(require(`../../activations/${this.activation}.glsl`))
      this.recurrentActivationProgram = webgl2.compileProgram(
        require(`../../activations/${this.recurrentActivation}.glsl`)
      )
      this.gateSummationProgram = webgl2.compileProgram(require('./gateSummation.glsl'))
      this.gateProductProgram = webgl2.compileProgram(require('./gateProduct.glsl'))
      this.timestepReadProgram = webgl2.compileProgram(require('./timestepRead.glsl'))
      this.timestepWriteProgram = webgl2.compileProgram(require('./timestepWrite.glsl'))
      this.updateProgram = webgl2.compileProgram(require('./LSTM.update.glsl'))
    }
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   *
   * W weight tensor is split into W_i, W_f, W_c, W_o
   *
   * U weight tensor is split into U_i, U_f, U_c, U_o
   *
   * b weight tensor is split into b_i, b_f, b_c, b_o (or create empty bias if this.use_bias is false)
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    super.setWeights(weightsArr)

    const shape_W = this.weights['kernel'].tensor.shape
    this.weights['W_i'] = new Tensor([], [shape_W[0], this.units])
    this.weights['W_f'] = new Tensor([], [shape_W[0], this.units])
    this.weights['W_c'] = new Tensor([], [shape_W[0], this.units])
    this.weights['W_o'] = new Tensor([], [shape_W[0], this.units])
    ops.assign(this.weights['W_i'].tensor, this.weights['kernel'].tensor.hi(shape_W[0], this.units).lo(0, 0))
    ops.assign(
      this.weights['W_f'].tensor,
      this.weights['kernel'].tensor.hi(shape_W[0], 2 * this.units).lo(0, this.units)
    )
    ops.assign(
      this.weights['W_c'].tensor,
      this.weights['kernel'].tensor.hi(shape_W[0], 3 * this.units).lo(0, 2 * this.units)
    )
    ops.assign(
      this.weights['W_o'].tensor,
      this.weights['kernel'].tensor.hi(shape_W[0], 4 * this.units).lo(0, 3 * this.units)
    )

    const shape_U = this.weights['recurrent_kernel'].tensor.shape
    this.weights['U_i'] = new Tensor([], [shape_U[0], this.units])
    this.weights['U_f'] = new Tensor([], [shape_U[0], this.units])
    this.weights['U_c'] = new Tensor([], [shape_U[0], this.units])
    this.weights['U_o'] = new Tensor([], [shape_U[0], this.units])
    ops.assign(this.weights['U_i'].tensor, this.weights['recurrent_kernel'].tensor.hi(shape_U[0], this.units).lo(0, 0))
    ops.assign(
      this.weights['U_f'].tensor,
      this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 2 * this.units).lo(0, this.units)
    )
    ops.assign(
      this.weights['U_c'].tensor,
      this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 3 * this.units).lo(0, 2 * this.units)
    )
    ops.assign(
      this.weights['U_o'].tensor,
      this.weights['recurrent_kernel'].tensor.hi(shape_U[0], 4 * this.units).lo(0, 3 * this.units)
    )

    this.weights['b_i'] = new Tensor([], [this.units])
    this.weights['b_f'] = new Tensor([], [this.units])
    this.weights['b_c'] = new Tensor([], [this.units])
    this.weights['b_o'] = new Tensor([], [this.units])
    if (this.use_bias) {
      ops.assign(this.weights['b_i'].tensor, this.weights['bias'].tensor.hi(this.units).lo(0))
      ops.assign(this.weights['b_f'].tensor, this.weights['bias'].tensor.hi(2 * this.units).lo(this.units))
      ops.assign(this.weights['b_c'].tensor, this.weights['bias'].tensor.hi(3 * this.units).lo(2 * this.units))
      ops.assign(this.weights['b_o'].tensor, this.weights['bias'].tensor.hi(4 * this.units).lo(3 * this.units))
    }

    if (this.gpu) {
      const names = ['W_i', 'W_f', 'W_c', 'W_o', 'U_i', 'U_f', 'U_c', 'U_o', 'b_i', 'b_f', 'b_c', 'b_o']
      names.forEach(name => {
        this.weights[name].createGLTexture()
      })
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
      this._callGPU(x)
    } else {
      this._callCPU(x)
    }
    return this.output
  }

  _combine = cwise({
    args: ['array', 'array', 'array', 'array'],
    body: function(_y, _x1, _x2, _b) {
      _y = _x1 + _x2 + _b
    }
  })

  _update = cwise({
    args: ['array', 'array', 'array', 'array'],
    body: function(_c, _ctm1, _i, _f) {
      _c = _c * _i + _ctm1 * _f
    }
  })

  /**
   * CPU call
   *
   * @param {Tensor} x
   */
  _callCPU(x) {
    const dimInputGate = this.weights['b_i'].tensor.shape[0]
    const dimCandidate = this.weights['b_c'].tensor.shape[0]
    const dimForgetGate = this.weights['b_f'].tensor.shape[0]
    const dimOutputGate = this.weights['b_o'].tensor.shape[0]

    const currentInputGateState = new Tensor([], [dimInputGate])
    const tempXI = new Tensor([], [dimInputGate])
    const tempHI = new Tensor([], [dimInputGate])

    const currentForgetGateState = new Tensor([], [dimForgetGate])
    const tempXF = new Tensor([], [dimForgetGate])
    const tempHF = new Tensor([], [dimForgetGate])

    const currentOutputGateState = new Tensor([], [dimOutputGate])
    const tempXO = new Tensor([], [dimOutputGate])
    const tempHO = new Tensor([], [dimOutputGate])

    const currentCandidate = new Tensor([], [dimCandidate])
    const tempXC = new Tensor([], [dimCandidate])
    const tempHC = new Tensor([], [dimCandidate])
    const previousCandidate =
      this.stateful && this.previousCandidate ? this.previousCandidate : new Tensor([], [dimCandidate])

    const currentHiddenState =
      this.stateful && this.currentHiddenState ? this.currentHiddenState : new Tensor([], [dimCandidate])
    const previousHiddenState = new Tensor([], [dimCandidate])

    this.hiddenStateSequence = new Tensor([], [x.tensor.shape[0], dimCandidate])

    const currentX = new Tensor([], [x.tensor.shape[1]])

    const _step = () => {
      ops.assign(previousHiddenState.tensor, currentHiddenState.tensor)

      gemv(1, this.weights['W_i'].tensor.transpose(1, 0), currentX.tensor, 1, tempXI.tensor)
      gemv(1, this.weights['U_i'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHI.tensor)
      this._combine(currentInputGateState.tensor, tempXI.tensor, tempHI.tensor, this.weights['b_i'].tensor)
      this.recurrentActivationFunc(currentInputGateState)

      gemv(1, this.weights['W_f'].tensor.transpose(1, 0), currentX.tensor, 1, tempXF.tensor)
      gemv(1, this.weights['U_f'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHF.tensor)
      this._combine(currentForgetGateState.tensor, tempXF.tensor, tempHF.tensor, this.weights['b_f'].tensor)
      this.recurrentActivationFunc(currentForgetGateState)

      gemv(1, this.weights['W_o'].tensor.transpose(1, 0), currentX.tensor, 1, tempXO.tensor)
      gemv(1, this.weights['U_o'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHO.tensor)
      this._combine(currentOutputGateState.tensor, tempXO.tensor, tempHO.tensor, this.weights['b_o'].tensor)
      this.recurrentActivationFunc(currentOutputGateState)

      gemv(1, this.weights['W_c'].tensor.transpose(1, 0), currentX.tensor, 1, tempXC.tensor)
      gemv(1, this.weights['U_c'].tensor.transpose(1, 0), previousHiddenState.tensor, 1, tempHC.tensor)
      this._combine(currentCandidate.tensor, tempXC.tensor, tempHC.tensor, this.weights['b_c'].tensor)
      this.activationFunc(currentCandidate)

      this._update(
        currentCandidate.tensor,
        previousCandidate.tensor,
        currentInputGateState.tensor,
        currentForgetGateState.tensor
      )

      ops.assign(previousCandidate.tensor, currentCandidate.tensor)

      this.activationFunc(currentCandidate)
      ops.mul(currentHiddenState.tensor, currentOutputGateState.tensor, currentCandidate.tensor)
    }

    for (let i = 0, len = x.tensor.shape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i
      ops.assign(currentX.tensor, x.tensor.pick(inputIndex, null))

      // clear temp tensors
      const tempTensors = [tempXI, tempHI, tempXF, tempHF, tempXO, tempHO, tempXC, tempHC]
      tempTensors.forEach(temp => ops.assigns(temp.tensor, 0))

      // advance timestep
      _step()

      ops.assign(this.hiddenStateSequence.tensor.pick(i, null), currentHiddenState.tensor)
    }

    if (this.returnSequences) {
      this.output = this.hiddenStateSequence
    } else {
      this.output = currentHiddenState
    }

    if (this.stateful) {
      this.previousCandidate = previousCandidate
      this.currentHiddenState = currentHiddenState
    }
  }

  /**
   * Advance time step in _callGPU
   */
  _stepGPU() {
    webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.previousHiddenState,
      inputs: [{ texture: this.currentHiddenState.glTexture, type: '2d', name: 'source' }]
    })

    // input gate

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXI,
      inputs: [
        { texture: this.currentX.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['W_i'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['W_i'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['W_i'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHI,
      inputs: [
        { texture: this.previousHiddenState.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['U_i'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['U_i'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['U_i'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentInputGateStatePreactiv,
      inputs: [
        { texture: this.tempXI.glTexture, type: '2d', name: 't1' },
        { texture: this.tempHI.glTexture, type: '2d', name: 't2' },
        { texture: this.weights['b_i'].glTexture, type: '2d', name: 'bias' }
      ]
    })

    if (this.recurrentActivation !== 'linear') {
      webgl2.runProgram({
        program: this.recurrentActivationProgram,
        output: this.currentInputGateState,
        inputs: [{ texture: this.currentInputGateStatePreactiv.glTexture, type: '2d', name: 'x' }]
      })
    } else {
      this.currentInputGateState = this.currentInputGateStatePreactiv
    }

    // forget gate

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXF,
      inputs: [
        { texture: this.currentX.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['W_f'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['W_f'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['W_f'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHF,
      inputs: [
        { texture: this.previousHiddenState.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['U_f'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['U_f'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['U_f'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentForgetGateStatePreactiv,
      inputs: [
        { texture: this.tempXF.glTexture, type: '2d', name: 't1' },
        { texture: this.tempHF.glTexture, type: '2d', name: 't2' },
        { texture: this.weights['b_f'].glTexture, type: '2d', name: 'bias' }
      ]
    })

    if (this.recurrentActivation !== 'linear') {
      webgl2.runProgram({
        program: this.recurrentActivationProgram,
        output: this.currentForgetGateState,
        inputs: [{ texture: this.currentForgetGateStatePreactiv.glTexture, type: '2d', name: 'x' }]
      })
    } else {
      this.currentForgetGateState = this.currentForgetGateStatePreactiv
    }

    // output gate

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXO,
      inputs: [
        { texture: this.currentX.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['W_o'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['W_o'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['W_o'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHO,
      inputs: [
        { texture: this.previousHiddenState.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['U_o'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['U_o'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['U_o'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentOutputGateStatePreactiv,
      inputs: [
        { texture: this.tempXO.glTexture, type: '2d', name: 't1' },
        { texture: this.tempHO.glTexture, type: '2d', name: 't2' },
        { texture: this.weights['b_o'].glTexture, type: '2d', name: 'bias' }
      ]
    })

    if (this.recurrentActivation !== 'linear') {
      webgl2.runProgram({
        program: this.recurrentActivationProgram,
        output: this.currentOutputGateState,
        inputs: [{ texture: this.currentOutputGateStatePreactiv.glTexture, type: '2d', name: 'x' }]
      })
    } else {
      this.currentOutputGateState = this.currentOutputGateStatePreactiv
    }

    // candidate

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempXC,
      inputs: [
        { texture: this.currentX.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['W_c'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['W_c'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['W_c'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.tempHC,
      inputs: [
        { texture: this.previousHiddenState.glTexture, type: '2d', name: 'A' },
        { texture: this.weights['U_c'].glTexture, type: '2d', name: 'B' }
      ],
      uniforms: [
        { value: 0, type: 'bool', name: 'addC' },
        { value: 1, type: 'int', name: 'M' },
        { value: this.weights['U_c'].glTextureShape[0], type: 'int', name: 'K' },
        { value: this.weights['U_c'].glTextureShape[1], type: 'int', name: 'N' }
      ]
    })

    webgl2.runProgram({
      program: this.gateSummationProgram,
      output: this.currentCandidatePreactiv,
      inputs: [
        { texture: this.tempXC.glTexture, type: '2d', name: 't1' },
        { texture: this.tempHC.glTexture, type: '2d', name: 't2' },
        { texture: this.weights['b_c'].glTexture, type: '2d', name: 'bias' }
      ]
    })

    if (this.activation !== 'linear') {
      webgl2.runProgram({
        program: this.activationProgram,
        output: this.currentCandidate,
        inputs: [{ texture: this.currentCandidatePreactiv.glTexture, type: '2d', name: 'x' }]
      })
    } else {
      this.currentCandidate = this.currentCandidatePreactiv
    }

    webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.currentCandidateCopy,
      inputs: [{ texture: this.currentCandidate.glTexture, type: '2d', name: 'source' }]
    })

    webgl2.runProgram({
      program: this.updateProgram,
      output: this.currentCandidate,
      inputs: [
        { texture: this.currentCandidateCopy.glTexture, type: '2d', name: 'c' },
        { texture: this.previousCandidate.glTexture, type: '2d', name: 'ctm1' },
        { texture: this.currentInputGateState.glTexture, type: '2d', name: 'i' },
        { texture: this.currentForgetGateState.glTexture, type: '2d', name: 'f' }
      ]
    })

    webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.previousCandidate,
      inputs: [{ texture: this.currentCandidate.glTexture, type: '2d', name: 'source' }]
    })

    webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.currentCandidatePreactiv,
      inputs: [{ texture: this.currentCandidate.glTexture, type: '2d', name: 'source' }]
    })

    if (this.activation !== 'linear') {
      webgl2.runProgram({
        program: this.activationProgram,
        output: this.currentCandidate,
        inputs: [{ texture: this.currentCandidatePreactiv.glTexture, type: '2d', name: 'x' }]
      })
    } else {
      this.currentCandidate = this.currentCandidatePreactiv
    }

    webgl2.runProgram({
      program: this.gateProductProgram,
      output: this.currentHiddenState,
      inputs: [
        { texture: this.currentOutputGateState.glTexture, type: '2d', name: 't1' },
        { texture: this.currentCandidate.glTexture, type: '2d', name: 't2' }
      ]
    })
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (!x.glTexture) {
      x.createGLTexture()
    }

    const dimInputGate = this.weights['b_i'].glTextureShape[1]
    const dimCandidate = this.weights['b_c'].glTextureShape[1]
    const dimForgetGate = this.weights['b_f'].glTextureShape[1]
    const dimOutputGate = this.weights['b_o'].glTextureShape[1]

    if (!this.currentInputGateState) {
      this.currentInputGateState = new Tensor([], [dimInputGate])
      this.currentInputGateState.createGLTexture()
    }
    if (!this.currentInputGateStatePreactiv) {
      this.currentInputGateStatePreactiv = new Tensor([], [dimInputGate])
      this.currentInputGateStatePreactiv.createGLTexture()
    }
    if (!this.tempXI) {
      this.tempXI = new Tensor([], [dimInputGate])
      this.tempXI.createGLTexture()
    }
    if (!this.tempHI) {
      this.tempHI = new Tensor([], [dimInputGate])
      this.tempHI.createGLTexture()
    }

    if (!this.currentForgetGateState) {
      this.currentForgetGateState = new Tensor([], [dimForgetGate])
      this.currentForgetGateState.createGLTexture()
    }
    if (!this.currentForgetGateStatePreactiv) {
      this.currentForgetGateStatePreactiv = new Tensor([], [dimForgetGate])
      this.currentForgetGateStatePreactiv.createGLTexture()
    }
    if (!this.tempXF) {
      this.tempXF = new Tensor([], [dimForgetGate])
      this.tempXF.createGLTexture()
    }
    if (!this.tempHF) {
      this.tempHF = new Tensor([], [dimForgetGate])
      this.tempHF.createGLTexture()
    }

    if (!this.currentOutputGateState) {
      this.currentOutputGateState = new Tensor([], [dimOutputGate])
      this.currentOutputGateState.createGLTexture()
    }
    if (!this.currentOutputGateStatePreactiv) {
      this.currentOutputGateStatePreactiv = new Tensor([], [dimOutputGate])
      this.currentOutputGateStatePreactiv.createGLTexture()
    }
    if (!this.tempXO) {
      this.tempXO = new Tensor([], [dimOutputGate])
      this.tempXO.createGLTexture()
    }
    if (!this.tempHO) {
      this.tempHO = new Tensor([], [dimOutputGate])
      this.tempHO.createGLTexture()
    }

    if (!this.currentCandidate) {
      this.currentCandidate = new Tensor([], [dimCandidate])
      this.currentCandidate.createGLTexture()
    }
    if (!this.currentCandidateCopy) {
      this.currentCandidateCopy = new Tensor([], [dimCandidate])
      this.currentCandidateCopy.createGLTexture()
    }
    if (!this.currentCandidatePreactiv) {
      this.currentCandidatePreactiv = new Tensor([], [dimCandidate])
      this.currentCandidatePreactiv.createGLTexture()
    }
    if (!this.tempXC) {
      this.tempXC = new Tensor([], [dimCandidate])
      this.tempXC.createGLTexture()
    }
    if (!this.tempHC) {
      this.tempHC = new Tensor([], [dimCandidate])
      this.tempHC.createGLTexture()
    }
    if (!this.previousCandidate || !this.stateful) {
      this.previousCandidate = new Tensor([], [dimCandidate])
      this.previousCandidate.createGLTexture()
    }

    if (!this.currentHiddenState || !this.stateful) {
      this.currentHiddenState = new Tensor([], [dimCandidate])
      this.currentHiddenState.createGLTexture()
    }
    if (!this.previousHiddenState) {
      this.previousHiddenState = new Tensor([], [dimCandidate])
      this.previousHiddenState.createGLTexture()
    }

    if (!this.hiddenStateSequence) {
      this.hiddenStateSequence = new Tensor([], [x.glTextureShape[0], dimCandidate])
      this.hiddenStateSequence.createGLTexture()
    }
    if (!this.hiddenStateSequenceCopy) {
      this.hiddenStateSequenceCopy = new Tensor([], [x.glTextureShape[0], dimCandidate])
      this.hiddenStateSequenceCopy.createGLTexture()
    }

    if (!this.currentX) {
      this.currentX = new Tensor([], [x.glTextureShape[1]])
      this.currentX.createGLTexture()
    }

    for (let i = 0, len = x.glTextureShape[0]; i < len; i++) {
      const inputIndex = this.goBackwards ? len - i - 1 : i

      webgl2.runProgram({
        program: this.timestepReadProgram,
        output: this.currentX,
        inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }],
        uniforms: [{ value: inputIndex, type: 'int', name: 'index' }]
      })

      this._stepGPU()

      if (this.returnSequences) {
        webgl2.runProgram({
          program: this.copyTextureProgram,
          output: this.hiddenStateSequenceCopy,
          inputs: [{ texture: this.hiddenStateSequence.glTexture, type: '2d', name: 'source' }]
        })
        webgl2.runProgram({
          program: this.timestepWriteProgram,
          output: this.hiddenStateSequence,
          inputs: [
            { texture: this.currentHiddenState.glTexture, type: '2d', name: 'x' },
            { texture: this.hiddenStateSequenceCopy.glTexture, type: '2d', name: 'y' }
          ],
          uniforms: [{ value: i, type: 'int', name: 'index' }]
        })
      }
    }

    if (this.returnSequences) {
      this.output = this.hiddenStateSequence
    } else {
      this.output = this.currentHiddenState
    }

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture()
    }
  }
}
