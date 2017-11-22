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
 * Cropping2D layer class
 */
class Cropping2D extends _Layer2.default {
  /**
   * Creates a Cropping2D layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number|number[]|number[][]} [attrs.cropping] - int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints
   * @param {string} [attrs.data_format] - either 'channels_last' or 'channels_first'
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Cropping2D';

    const { cropping = [[0, 0], [0, 0]], data_format = 'channels_last' } = attrs;

    if (Array.isArray(cropping)) {
      if (Array.isArray(cropping[0])) {
        // [[int, int], [int, int]]
        this.cropping = cropping;
      } else {
        // [int, int]
        this.cropping = [[cropping[0], cropping[0]], [cropping[1], cropping[1]]];
      }
    } else {
      // int
      this.cropping = [[cropping, cropping], [cropping, cropping]];
    }

    this.dataFormat = data_format;

    // GPU setup
    if (this.gpu) {
      this.mapInputProgram = _WebGL.webgl2.compileProgram(require('../../mapInput.glsl'));
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
    // convert to channels_last ordering if necessary
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(1, 2, 0);
    }

    this.inputShape = x.tensor.shape;
    this.outputShape = [this.inputShape[0] - this.cropping[0][0] - this.cropping[0][1], this.inputShape[1] - this.cropping[1][0] - this.cropping[1][1], this.inputShape[2]];
    this.output = new _Tensor2.default([], this.outputShape);
    _ndarrayOps2.default.assign(this.output.tensor, x.tensor.hi(this.inputShape[0] - this.cropping[0][1], this.inputShape[1] - this.cropping[1][1], this.inputShape[2]).lo(this.cropping[0][0], this.cropping[1][0], 0));

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'channels_first') {
      x.tensor = x.tensor.transpose(2, 0, 1);
      this.output.tensor = this.output.tensor.transpose(2, 0, 1);
    }
  }

  /**
   * Creates row/col index mappings to map input texture to output texture
   *
   * @param {Object} indicesForReshaped
   */
  _createIndexMap(indicesForReshaped) {
    if (this.rowIndexMap && this.colIndexMap) {
      return;
    }

    const indicesRow = new _Tensor2.default(indicesForReshaped.row.data, indicesForReshaped.row.shape, { type: Int32Array });
    const indicesCol = new _Tensor2.default(indicesForReshaped.col.data, indicesForReshaped.col.shape, { type: Int32Array });

    this.rowIndexMap = new _Tensor2.default([], this.outputShape, { type: Int32Array });
    this.colIndexMap = new _Tensor2.default([], this.outputShape, { type: Int32Array });

    const sliceStart = this.dataFormat === 'channels_first' ? [0, this.cropping[0][0], this.cropping[1][0]] : [this.cropping[0][0], this.cropping[1][0], 0];
    const sliceEnd = this.dataFormat === 'channels_first' ? [this.inputShape[0], this.inputShape[1] - this.cropping[0][1], this.inputShape[2] - this.cropping[1][1]] : [this.inputShape[0] - this.cropping[0][1], this.inputShape[1] - this.cropping[1][1], this.inputShape[2]];

    _ndarrayOps2.default.assign(this.rowIndexMap.tensor, indicesRow.tensor.hi(...sliceEnd).lo(...sliceStart));
    _ndarrayOps2.default.assign(this.colIndexMap.tensor, indicesCol.tensor.hi(...sliceEnd).lo(...sliceStart));

    this.rowIndexMap.reshapeTo2DSquare();
    this.colIndexMap.reshapeTo2DSquare();
    this.rowIndexMap.createGLTexture('2d', 'int');
    this.colIndexMap.createGLTexture('2d', 'int');
  }

  /**
  * GPU call
  *
  * @param {Tensor} x
  */
  _callGPU(x) {
    if (!x.glTexture) {
      x.reshapeTo2DSquare();
      x.createGLTexture();
    }
    this.inputShape = x.originalShape;
    this.outputShape = this.dataFormat === 'channels_first' ? [this.inputShape[0], this.inputShape[1] - this.cropping[0][0] - this.cropping[0][1], this.inputShape[2] - this.cropping[1][0] - this.cropping[1][1]] : [this.inputShape[0] - this.cropping[0][0] - this.cropping[0][1], this.inputShape[1] - this.cropping[1][0] - this.cropping[1][1], this.inputShape[2]];

    this._createIndexMap(x.indicesForReshaped);

    if (!this.output) {
      this.output = new _Tensor2.default([], this.outputShape);
      this.output.reshapeTo2DSquare();
      this.output.createGLTexture();
    }

    _WebGL.webgl2.runProgram({
      program: this.mapInputProgram,
      output: this.output,
      inputs: [{ texture: x.glTexture, type: '2d', name: 'x' }, { texture: this.rowIndexMap.glTexture, type: '2d', name: 'rowIndexMap' }, { texture: this.colIndexMap.glTexture, type: '2d', name: 'colIndexMap' }]
    });

    // GPU -> CPU data transfer
    if (this.outbound.length === 0) {
      this.output.transferFromGLTexture();
      this.output.reshapeFrom2DSquare();
    }
  }
}
exports.default = Cropping2D;