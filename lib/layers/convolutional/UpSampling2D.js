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
 * UpSampling2D layer class
 */
class UpSampling2D extends _Layer2.default {
  /**
   * Creates a UpSampling2D layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number|number[]} [attrs.size] - upsampling factor, int or tuple of int (length 2)
   * @param {string} [attrs.data_format] - either 'channels_last' or 'channels_first'
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'UpSampling2D';

    const { size = [2, 2], data_format = 'channels_last' } = attrs;

    if (Array.isArray(size)) {
      this.size = size;
    } else {
      this.size = [size, size];
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
    this.outputShape = [this.inputShape[0] * this.size[0], this.inputShape[1] * this.size[1], this.inputShape[2]];
    this.output = new _Tensor2.default([], this.outputShape);
    for (let i = 0; i < this.size[0]; i++) {
      for (let j = 0; j < this.size[1]; j++) {
        _ndarrayOps2.default.assign(this.output.tensor.lo(i, j, 0).step(this.size[0], this.size[1], 1), x.tensor);
      }
    }

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

    for (let i = 0; i < this.size[0]; i++) {
      for (let j = 0; j < this.size[1]; j++) {
        const sliceStart = this.dataFormat === 'channels_first' ? [0, i, j] : [i, j, 0];
        const step = this.dataFormat === 'channels_first' ? [1, this.size[0], this.size[1]] : [this.size[0], this.size[1], 1];
        _ndarrayOps2.default.assign(this.rowIndexMap.tensor.lo(...sliceStart).step(...step), indicesRow.tensor);
        _ndarrayOps2.default.assign(this.colIndexMap.tensor.lo(...sliceStart).step(...step), indicesCol.tensor);
      }
    }

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
    this.outputShape = this.dataFormat === 'channels_first' ? [this.inputShape[0], this.inputShape[1] * this.size[0], this.inputShape[2] * this.size[1]] : [this.inputShape[0] * this.size[0], this.inputShape[1] * this.size[1], this.inputShape[2]];

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
exports.default = UpSampling2D;