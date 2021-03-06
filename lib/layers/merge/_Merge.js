'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _isEqual = require('lodash/isEqual');

var _isEqual2 = _interopRequireDefault(_isEqual);

var _range = require('lodash/range');

var _range2 = _interopRequireDefault(_range);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * _Merge layer class
 */
class _Merge extends _Layer2.default {
  /**
   * Creates a _Merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = '_Merge';
    this.isMergeLayer = true;

    // GPU setup
    if (this.gpu) {
      this.copyTextureProgram = _WebGL.webgl2.compileProgram(require('../../copyTexture.glsl'));
    }
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor[]} inputs
   * @returns {Tensor}
   */
  call(inputs) {
    if (this.gpu) {
      inputs.forEach(input => {
        if (!input.glTexture) {
          input.createGLTexture();
        }
      });
      this._callGPU(inputs);
    } else {
      const valid = this._validateInputs(inputs);
      if (!valid) {
        this.throwError('Invalid inputs to call method.');
      }
      this._callCPU(inputs);
    }
    return this.output;
  }

  /**
   * Internal method for validating inputs
   *
   * @param {Tensor[]} inputs
   * @returns {boolean}
   */
  _validateInputs(inputs) {
    const shapes = inputs.map(x => x.tensor.shape.slice());
    if (['sum', 'diff', 'mul', 'ave', 'max', 'min'].indexOf(this.mode) > -1) {
      if (!shapes.every(shape => (0, _isEqual2.default)(shape, shapes[0]))) {
        this.throwError(`All input shapes must be the same for mode ${this.mode}.`);
      }
    }
    if (this.mode === 'dot') {
      if (inputs.length !== 2) {
        this.throwError(`Exactly 2 inputs required for mode ${this.mode}.`);
      }
      if (this.dotAxes[0] < 0) {
        this.dotAxes[0] = shapes[0].length + this.dotAxes[0];
      }
      if (this.dotAxes[1] < 0) {
        this.dotAxes[1] = shapes[1].length + this.dotAxes[1];
      }
      if (shapes[0][this.dotAxes[0]] !== shapes[1][this.dotAxes[1]]) {
        this.throwError('Dimensions incompatibility using dot mode.');
      }
    } else if (this.mode === 'concat') {
      let nonConcatShapes = shapes.slice();
      let _concatAxis = this.concatAxis < 0 ? nonConcatShapes[0].length + this.concatAxis : this.concatAxis;
      if (this.concatAxis === 0) _concatAxis = 0;
      (0, _range2.default)(nonConcatShapes.length).forEach(i => {
        nonConcatShapes[i].splice(_concatAxis, 1);
      });
      if (!nonConcatShapes.every(shape => (0, _isEqual2.default)(shape, nonConcatShapes[0]))) {
        this.throwError('In concat mode, all shapes must be the same except along the concat axis.');
      }
    }
    return true;
  }

  /**
   * CPU call
   *
   * implemented in child classes
   *
   * @param {Tensor[]} inputs
   */
  _callCPU(inputs) {}

  /**
   * GPU call
   *
   * mode: sum, diff, mul, ave, max, min
   *
   * method for mode concat/dot implemented in child class
   *
   * @param {Tensor[]} inputs
   */
  _callGPU(inputs) {
    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new _Tensor2.default([], inputs[0].glTextureShape);
      this.output.createGLTexture();
      if (inputs[0].is1D) {
        this.output.is1D = inputs[0].is1D;
      } else if (inputs[0].is2DReshaped) {
        this.output.is2DReshaped = inputs[0].is2DReshaped;
        this.output.originalShape = inputs[0].originalShape;
        this.output.indicesForReshaped = inputs[0].indicesForReshaped;
      }
    }

    const numInputs = inputs.length;

    const mergeUniforms = [];
    if (this.mode === 'ave') {
      mergeUniforms.push({ value: numInputs, type: 'int', name: 'numInputs' });
    }
    _WebGL.webgl2.runProgram({
      program: this.mergeProgram,
      output: this.output,
      inputs: [{ texture: inputs[0].glTexture, type: '2d', name: 'input1' }, { texture: inputs[1].glTexture, type: '2d', name: 'input2' }],
      uniforms: mergeUniforms
    });

    if (numInputs > 2) {
      if (!this.runningOutput) {
        this.runningOutput = new _Tensor2.default([], inputs[0].glTextureShape);
        this.runningOutput.createGLTexture();
      }
      if (this.mode === 'ave') {
        mergeUniforms.push({ value: 1, type: 'bool', name: 'additional' });
      }

      for (let i = 2; i < numInputs; i++) {
        // copy output texture to intermediate output
        _WebGL.webgl2.runProgram({
          program: this.copyTextureProgram,
          output: this.runningOutput,
          inputs: [{ texture: this.output.glTexture, type: '2d', name: 'source' }]
        });

        _WebGL.webgl2.runProgram({
          program: this.mergeProgram,
          output: this.output,
          inputs: [{ texture: this.runningOutput.glTexture, type: '2d', name: 'input1' }, { texture: inputs[i].glTexture, type: '2d', name: 'input2' }],
          uniforms: mergeUniforms
        });
      }
    }

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
      if (this.output.is2DReshaped) {
        this.output.reshapeFrom2D();
      }
    }
  }
}
exports.default = _Merge;