'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _last2 = require('lodash/last');

var _last3 = _interopRequireDefault(_last2);

var _range2 = require('lodash/range');

var _range3 = _interopRequireDefault(_range2);

var _isEqual2 = require('lodash/isEqual');

var _isEqual3 = _interopRequireDefault(_isEqual2);

var _Layer = require('../../Layer');

var _Layer2 = _interopRequireDefault(_Layer);

var _Tensor = require('../../Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('../../WebGL2');

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Permute layer class
 * Note there is no concept of batch size in these layers (single-batch), so dim numbers 1 less
 * i.e., dim 1 in keras corresponds to dim 0 here, etc.
 */
class Permute extends _Layer2.default {
  /**
   * Creates a Permute layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number[]} [attrs.dims]
   */
  constructor(attrs = {}) {
    super(attrs);
    this.layerClass = 'Permute';

    const { dims = [] } = attrs;
    this.dims = dims.map(dim => dim - 1);

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
    if (x.tensor.shape.length <= 1 || (0, _isEqual3.default)((0, _range3.default)(x.tensor.shape.length), this.dims)) {
      this.output = x;
      return this.output;
    }

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
    if (this.dims.length !== x.tensor.shape.length) {
      this.throwError('The specified dims permutation must match the number of dimensions.');
    }

    const outputShape = this.dims.map(i => x.tensor.shape[i]);
    this.output = new _Tensor2.default([], outputShape);
    _ndarrayOps2.default.assign(this.output.tensor, x.tensor.transpose(...this.dims));
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

    if (this.inputShape.length === 2) {
      for (let i = 0; i < this.inputShape[0]; i++) {
        _ndarrayOps2.default.assigns(indicesRow.tensor.pick(i, null), i);
      }
    } else if (this.inputShape.length === 3) {
      for (let i = 0; i < this.inputShape[0]; i++) {
        for (let j = 0; j < this.inputShape[1]; j++) {
          _ndarrayOps2.default.assigns(indicesRow.tensor.pick(i, j, null), i * this.inputShape[1] + j);
        }
      }
    } else if (this.inputShape.length === 4) {
      for (let i = 0; i < this.inputShape[0]; i++) {
        for (let j = 0; j < this.inputShape[1]; j++) {
          for (let k = 0; k < this.inputShape[2]; k++) {
            _ndarrayOps2.default.assigns(indicesRow.tensor.pick(i, j, k, null), i * this.inputShape[1] * this.inputShape[2] + j * this.inputShape[2] + k);
          }
        }
      }
    }
    for (let c = 0; c < (0, _last3.default)(this.inputShape); c++) {
      _ndarrayOps2.default.assigns(indicesCol.tensor.pick(...Array(this.inputShape.length - 1).fill(null), c), c);
    }

    const outputShape = this.dims.map(i => this.inputShape[i]);
    this.rowIndexMap = new _Tensor2.default([], outputShape, { type: Int32Array });
    this.colIndexMap = new _Tensor2.default([], outputShape, { type: Int32Array });
    _ndarrayOps2.default.assign(this.rowIndexMap.tensor, indicesRow.tensor.transpose(...this.dims));
    _ndarrayOps2.default.assign(this.colIndexMap.tensor, indicesCol.tensor.transpose(...this.dims));
    if (outputShape.length > 2) {
      this.rowIndexMap.reshapeTo2D();
      this.colIndexMap.reshapeTo2D();
    }

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
      this.inputShape = x.tensor.shape;
      if (x.tensor.shape.length <= 2) {
        x.createGLTexture();
      } else if (x.tensor.shape.length > 2 && !x.is2DReshaped) {
        x.reshapeTo2D();
        x.createGLTexture();
      }
    } else if (x.is2DReshaped) {
      this.inputShape = x.originalShape;
    } else {
      this.inputShape = x.tensor.shape;
    }
    this._createIndexMap();

    if (!this.output) {
      const outputShape = this.dims.map(i => this.inputShape[i]);
      this.output = new _Tensor2.default([], outputShape);
      if (outputShape.length > 2) {
        this.output.reshapeTo2D();
      }
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
      if (this.output.is2DReshaped) {
        this.output.reshapeFrom2D(this.axis);
      }
    }
  }
}
exports.default = Permute;