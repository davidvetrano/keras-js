'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _Merge2 = require('./_Merge');

var _Merge3 = _interopRequireDefault(_Merge2);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _tensorUtils = require('../../utils/tensorUtils');

var tensorUtils = _interopRequireWildcard(_tensorUtils);

var _ndarrayConcatRows = require('ndarray-concat-rows');

var _ndarrayConcatRows2 = _interopRequireDefault(_ndarrayConcatRows);

var _sum = require('lodash/sum');

var _sum2 = _interopRequireDefault(_sum);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Concatenate merge layer class, extends abstract _Merge class
 */
class Concatenate extends _Merge3.default {
  /**
   * Creates a Concatenate merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Concatenate';

    this.mode = 'concat';

    const { axis = -1 } = attrs;

    // no mini-batch axis here, so we subtract 1 if given axis > 0
    this.concatAxis = axis <= 0 ? axis : axis - 1;

    // GPU setup
    if (this.gpu) {
      this.mergeProgram = _WebGL.webgl2.compileProgram(require('./Concatenate.glsl'));
    }
  }

  /**
   * CPU call
   *
   * @param {Tensor[]} inputs
   */
  _callCPU(inputs) {
    const outputShape = inputs[0].tensor.shape.slice();
    const _concatAxis = this.concatAxis < 0 ? outputShape.length + this.concatAxis : this.concatAxis;
    inputs.slice(1, inputs.length).forEach(x => {
      const d = x.tensor.shape.slice()[_concatAxis];
      outputShape[_concatAxis] += d;
    });
    this.output = new _Tensor2.default([], outputShape);

    if (_concatAxis === 0) {
      (0, _ndarrayConcatRows2.default)(this.output.tensor, inputs.map(x => x.tensor));
    } else {
      let dimsAxisSwap = [_concatAxis];
      for (let i = 0; i < inputs[0].tensor.shape.length; i++) {
        if (i !== _concatAxis) dimsAxisSwap.push(i);
      }
      (0, _ndarrayConcatRows2.default)(this.output.tensor.transpose(...dimsAxisSwap), inputs.map(x => x.tensor.transpose(...dimsAxisSwap)));
    }
  }

  /**
   * GPU call
   *
   * @param {Tensor[]} inputs
   */
  _callGPU(inputs) {
    const outputShape = inputs[0].glTextureShape.slice();
    let _concatAxis = 1;
    if (inputs[0].is2DReshaped) {
      if (this.concatAxis === -1 || this.concatAxis === inputs[0].originalShape.length - 1) {
        _concatAxis = 1;
      } else {
        this.throwError('specified axis not supported for now.');
      }
    } else {
      if (this.concatAxis === -1 || this.concatAxis === 1) {
        _concatAxis = 1;
      } else if (this.concatAxis === -2 || this.concatAxis === 0) {
        _concatAxis = 0;
      } else {
        this.throwError('specified axis not supported for now.');
      }
    }

    // create output textures if doesn't already exist
    if (!this.output) {
      outputShape[_concatAxis] = (0, _sum2.default)(inputs.map(input => input.glTextureShape[_concatAxis]));
      this.output = new _Tensor2.default([], outputShape);
      this.output.createGLTexture();
      if (inputs[0].is1D) {
        this.output.is1D = inputs[0].is1D;
      } else if (inputs[0].is2DReshaped) {
        this.output.is2DReshaped = inputs[0].is2DReshaped;
        this.output.originalShape = inputs[0].originalShape;
        const _concatAxis = this.concatAxis < 0 ? this.output.originalShape.length + this.concatAxis : this.concatAxis;
        this.output.originalShape[_concatAxis] = (0, _sum2.default)(inputs.map(input => input.originalShape[_concatAxis]));
        this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.output.originalShape, false, _concatAxis);
      }
    }
    if (!this.runningOutput) {
      this.runningOutput = new _Tensor2.default([], outputShape);
      this.runningOutput.createGLTexture();
    }

    const numInputs = inputs.length;

    let offsetStart = 0;
    let offsetEnd = inputs[0].glTextureShape[_concatAxis];
    for (let i = 0; i < numInputs; i++) {
      // copy output texture to intermediate output
      _WebGL.webgl2.runProgram({
        program: this.copyTextureProgram,
        output: this.runningOutput,
        inputs: [{ texture: this.output.glTexture, type: '2d', name: 'source' }]
      });

      // run merge program
      _WebGL.webgl2.runProgram({
        program: this.mergeProgram,
        output: this.output,
        inputs: [{ texture: this.runningOutput.glTexture, type: '2d', name: 'runningOutput' }, { texture: inputs[i].glTexture, type: '2d', name: 'input1' }],
        uniforms: [{ value: this.output.glTextureShape[0], type: 'int', name: 'rows' }, { value: this.output.glTextureShape[1], type: 'int', name: 'cols' }, { value: _concatAxis, type: 'int', name: 'concatAxis' }, { value: offsetStart, type: 'int', name: 'offsetStart' }, { value: offsetEnd, type: 'int', name: 'offsetEnd' }]
      });

      if (i < numInputs - 1) {
        offsetStart += inputs[i].glTextureShape[_concatAxis];
        offsetEnd += inputs[i + 1].glTextureShape[_concatAxis];
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
exports.default = Concatenate;