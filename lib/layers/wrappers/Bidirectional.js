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

var _recurrent = require('../recurrent');

var recurrentLayers = _interopRequireWildcard(_recurrent);

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Bidirectional wrapper layer class
 */
class Bidirectional extends _Layer2.default {
  /**
   * Creates a Bidirectional wrapper layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {string} [attrs.merge_mode] - merge mode of component layers
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Bidirectional';

    const { layer, merge_mode = 'concat' } = attrs;

    if (!layer) {
      this.throwError('wrapped layer is undefined.');
    }
    if (!['SimpleRNN', 'GRU', 'LSTM'].includes(layer.class_name)) {
      this.throwError(`cannot wrap ${layer.class_name} layer.`);
    }
    if (!['concat', 'sum', 'mul', 'ave'].includes(merge_mode)) {
      this.throwError(`merge_mode ${merge_mode} not supported.`);
    }

    const forwardLayerAttrs = Object.assign({}, layer.config, { gpu: attrs.gpu });
    const backwardLayerAttrs = Object.assign({}, layer.config, { gpu: attrs.gpu });
    backwardLayerAttrs.go_backwards = !backwardLayerAttrs.go_backwards;
    this.forwardLayer = new recurrentLayers[layer.class_name](forwardLayerAttrs);
    this.backwardLayer = new recurrentLayers[layer.class_name](backwardLayerAttrs);

    // prevent GPU -> CPU data transfer by specifying non-empty outbound nodes array on internal layers
    this.forwardLayer.outbound = [null];
    this.backwardLayer.outbound = [null];

    this.mergeMode = merge_mode;
    this.returnSequences = layer.config.return_sequences;

    // GPU setup
    if (this.gpu) {
      this.copyTextureProgram = _WebGL.webgl2.compileProgram(require('../../copyTexture.glsl'));
      if (this.mergeMode === 'concat') {
        this.mergeProgram = _WebGL.webgl2.compileProgram(require('./Bidirectional.concat.glsl'));
      } else if (this.mergeMode === 'sum') {
        this.mergeProgram = _WebGL.webgl2.compileProgram(require('./Bidirectional.sum.glsl'));
      } else if (this.mergeMode === 'mul') {
        this.mergeProgram = _WebGL.webgl2.compileProgram(require('./Bidirectional.mul.glsl'));
      } else if (this.mergeMode === 'ave') {
        this.mergeProgram = _WebGL.webgl2.compileProgram(require('./Bidirectional.ave.glsl'));
      }
    }
  }

  /**
   * Method for setting layer weights - passes weights to the wrapped layer
   *
   * Here, the weights array is concatenated from the forward layer and the backward layer
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  setWeights(weightsArr) {
    this.forwardLayer.setWeights(weightsArr.slice(0, weightsArr.length / 2));
    this.backwardLayer.setWeights(weightsArr.slice(weightsArr.length / 2));
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
    this.forwardLayer._callCPU(new _Tensor2.default(x.tensor.data, x.tensor.shape));
    this.backwardLayer._callCPU(new _Tensor2.default(x.tensor.data, x.tensor.shape));
    const forwardOutput = this.forwardLayer.output;
    const backwardOutput = this.backwardLayer.output;

    // when returnSequences = true, reverse results of backwardLayer
    if (this.returnSequences) {
      backwardOutput.tensor = backwardOutput.tensor.step(-1);
    }

    const outShape = forwardOutput.tensor.shape.slice();
    if (this.mergeMode === 'concat') {
      outShape[outShape.length - 1] += backwardOutput.tensor.shape[outShape.length - 1];
    }
    this.output = new _Tensor2.default([], outShape);

    if (this.mergeMode === 'concat') {
      if (this.returnSequences) {
        _ndarrayOps2.default.assign(this.output.tensor.hi(outShape[0], forwardOutput.tensor.shape[1]).lo(0, 0), forwardOutput.tensor);
        _ndarrayOps2.default.assign(this.output.tensor.hi(outShape[0], outShape[1]).lo(0, forwardOutput.tensor.shape[1]), backwardOutput.tensor);
      } else {
        _ndarrayOps2.default.assign(this.output.tensor.hi(forwardOutput.tensor.shape[0]).lo(0), forwardOutput.tensor);
        _ndarrayOps2.default.assign(this.output.tensor.hi(outShape[0]).lo(forwardOutput.tensor.shape[0]), backwardOutput.tensor);
      }
    } else if (this.mergeMode === 'sum') {
      _ndarrayOps2.default.addeq(this.output.tensor, forwardOutput.tensor);
      _ndarrayOps2.default.addeq(this.output.tensor, backwardOutput.tensor);
    } else if (this.mergeMode === 'mul') {
      _ndarrayOps2.default.assigns(this.output.tensor, 1);
      _ndarrayOps2.default.muleq(this.output.tensor, forwardOutput.tensor);
      _ndarrayOps2.default.muleq(this.output.tensor, backwardOutput.tensor);
    } else if (this.mergeMode === 'ave') {
      _ndarrayOps2.default.addeq(this.output.tensor, forwardOutput.tensor);
      _ndarrayOps2.default.addeq(this.output.tensor, backwardOutput.tensor);
      _ndarrayOps2.default.divseq(this.output.tensor, 2);
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
    if (!this.inputCopy) {
      this.inputCopy = new _Tensor2.default([], x.glTextureShape);
      this.inputCopy.createGLTexture();
    }

    _WebGL.webgl2.runProgram({
      program: this.copyTextureProgram,
      output: this.inputCopy,
      inputs: [{ texture: x.glTexture, type: '2d', name: 'source' }]
    });

    // run internal component layers
    this.forwardLayer._callGPU(x);
    this.backwardLayer._callGPU(this.inputCopy);
    const forwardOutput = this.forwardLayer.output;
    const backwardOutput = this.backwardLayer.output;

    const outShape = forwardOutput.glTextureShape.slice();
    if (this.mergeMode === 'concat') {
      outShape[1] += backwardOutput.glTextureShape[1];
    }
    if (!this.output) {
      this.output = new _Tensor2.default([], outShape);
      this.output.createGLTexture();
      if (!this.returnSequences) {
        this.output.is1D = true;
      }
    }

    // merge forward and backward outputs
    _WebGL.webgl2.runProgram({
      program: this.mergeProgram,
      output: this.output,
      inputs: [{ texture: forwardOutput.glTexture, type: '2d', name: 'forward' }, { texture: backwardOutput.glTexture, type: '2d', name: 'backward' }]
    });

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
    }
  }
}
exports.default = Bidirectional;