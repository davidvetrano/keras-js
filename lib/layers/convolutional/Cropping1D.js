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
 * Cropping1D layer class
 */
class Cropping1D extends _Layer2.default {
  /**
   * Creates a Cropping1D layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number|number[]} [attrs.cropping] - int or tuple of int (length 2)
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Cropping1D';

    const { cropping = [0, 0] } = attrs;

    if (Array.isArray(cropping)) {
      this.cropping = cropping;
    } else {
      this.cropping = [cropping, cropping];
    }

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
    this.inputShape = x.tensor.shape;
    this.outputShape = [this.inputShape[0] - this.cropping[0] - this.cropping[1], this.inputShape[1]];
    this.output = new _Tensor2.default([], this.outputShape);
    _ndarrayOps2.default.assign(this.output.tensor, x.tensor.hi(this.inputShape[0] - this.cropping[1], this.inputShape[2]).lo(this.cropping[0], 0));
  }

  /**
   * Creates row/col index mappings to map input texture to output texture
   */
  _createIndexMap() {
    if (this.rowIndexMap && this.colIndexMap) {
      return;
    }

    const indicesRow = new _Tensor2.default([], this.inputShape, { type: Int32Array });
    const indicesCol = new _Tensor2.default([], this.inputShape, { type: Int32Array });
    for (let i = 0; i < this.inputShape[0]; i++) {
      _ndarrayOps2.default.assigns(indicesRow.tensor.pick(i, null), i);
    }
    for (let j = 0; j < this.inputShape[1]; j++) {
      _ndarrayOps2.default.assigns(indicesCol.tensor.pick(null, j), j);
    }

    this.rowIndexMap = new _Tensor2.default([], this.outputShape, { type: Int32Array });
    this.colIndexMap = new _Tensor2.default([], this.outputShape, { type: Int32Array });
    const sliceStart = [this.cropping[0], 0];
    const sliceEnd = [this.inputShape[0] - this.cropping[1], this.inputShape[2]];
    _ndarrayOps2.default.assign(this.rowIndexMap.tensor, indicesRow.tensor.hi(...sliceEnd).lo(...sliceStart));
    _ndarrayOps2.default.assign(this.colIndexMap.tensor, indicesCol.tensor.hi(...sliceEnd).lo(...sliceStart));

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
      x.createGLTexture();
    }

    this.inputShape = x.tensor.shape;
    this.outputShape = [this.inputShape[0] - this.cropping[0] - this.cropping[1], this.inputShape[1]];
    this._createIndexMap();

    if (!this.output) {
      this.output = new _Tensor2.default([], this.outputShape);
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
    }
  }
}
exports.default = Cropping1D;