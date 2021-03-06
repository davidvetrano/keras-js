'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _keys2 = require('lodash/keys');

var _keys3 = _interopRequireDefault(_keys2);

var _isEqual2 = require('lodash/isEqual');

var _isEqual3 = _interopRequireDefault(_isEqual2);

var _map2 = require('lodash/map');

var _map3 = _interopRequireDefault(_map2);

var _every2 = require('lodash/every');

var _every3 = _interopRequireDefault(_every2);

var _find2 = require('lodash/find');

var _find3 = _interopRequireDefault(_find2);

var _sum2 = require('lodash/sum');

var _sum3 = _interopRequireDefault(_sum2);

var _values2 = require('lodash/values');

var _values3 = _interopRequireDefault(_values2);

var _bluebird = require('bluebird');

var _bluebird2 = _interopRequireDefault(_bluebird);

var _axios = require('axios');

var _axios2 = _interopRequireDefault(_axios);

var _layers = require('./layers');

var layers = _interopRequireWildcard(_layers);

var _Tensor = require('./Tensor');

var _Tensor2 = _interopRequireDefault(_Tensor);

var _WebGL = require('./WebGL2');

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

const axiosSource = _axios2.default.CancelToken.source();

/**
 * Model class
 */
class Model {
  /**
   * Create new Model class
   *
   * @param {Object} config.filepaths
   * @param {string} config.filepaths.modelFilepath - path to model architecture configuration (json)
   * @param {string} config.filepaths.weightsFilepath - path to weights data (arraybuffer)
   * @param {string} config.filepaths.metadataFilepath - path to weights metadata (json)
   * @param {Object} [config.headers] - any additional HTTP headers required for resource fetching
   * @param {boolean} [config.gpu] - enable GPU
   * @param {boolean} [config.layerCallPauses] - force next tick after each layer call
   */
  constructor(config = {}) {
    const {
      filepaths = {},
      headers = {},
      filesystem = false,
      gpu = false,
      transferLayerOutputs = false,
      layerCallPauses = false
    } = config;

    if (!filepaths.model || !filepaths.weights || !filepaths.metadata) {
      throw new Error('[Model] File paths must be declared for model, weights, and metadata.');
    }
    this.filepaths = filepaths;
    this.filetypes = { model: 'json', weights: 'arraybuffer', metadata: 'json'

      // HTTP(S) headers used during data fetching
    };this.headers = headers;

    // specifies that data files are from local file system
    // only in node
    this.filesystem = typeof window !== 'undefined' ? false : filesystem;

    this.data = {
      // object representing the model architecture configuration,
      // directly from the to_json() method in Keras
      model: {},
      // ArrayBuffer of all the weights, sequentially concatenated
      // see encoder.py for construction details - essentially the raw flattened
      // numerical data from the HDF5 file is extracted sequentially and concatenated.
      weights: null,
      // array of weight tensor metadata, used to reconstruct tensors from the raw
      // weights ArrayBuffer above.
      metadata: []

      // data request progress
    };this.dataRequestProgress = { model: 0, weights: 0, metadata: 0

      // flag to enable GPU where possible (disable in node environment)
    };this.gpu = typeof window !== 'undefined' && _WebGL.webgl2.isSupported ? gpu : false;

    // in GPU mode, transfer intermediate outputs of each layer from GPU->CPU (warning: decreases performance)
    this.transferLayerOutputs = transferLayerOutputs;

    // flag to enable 0 ms pauses after layer computation calls
    this.layerCallPauses = layerCallPauses;

    // map of model layers
    this.modelLayersMap = new Map();

    // map of input tensors
    this.inputTensorsMap = new Map();

    // names of input and output layers
    this.inputLayerNames = [];
    this.outputLayerNames = [];

    // array of model layer names with finished output
    this.finishedLayerNames = [];

    // flag while computations are being performed
    this.isRunning = false;

    // Promise for when Model class is initialized
    this._ready = this._initialize();
  }

  /**
   * Checks whether WebGL 2 is supported by browser
   */
  checkGPUSupport() {
    return _WebGL.webgl2.isSupported;
  }

  /**
   * Promise for when model data is loaded and layers are initialized.
   *
   * @returns {Promise}
   */
  ready() {
    return this._ready;
  }

  /**
   * Cancels any existing data requests
   */
  _interrupt() {
    axiosSource.cancel();
  }

  /**
   * Model initialization
   *
   * @returns {Promise}
   */
  async _initialize() {
    const dataTypes = ['model', 'weights', 'metadata'];
    const dataRequests = dataTypes.map(type => {
      return this.filesystem ? this._dataRequestFS(type) : this._dataRequestHTTP(type, this.headers);
    });

    try {
      await _bluebird2.default.all(dataRequests);
    } catch (err) {
      console.log(err);
      this._interrupt();
    }

    // build directed acyclic graph
    this._buildDAG();

    // run predict once with initial empty input tensors to cache variables such as shape inference
    // make sure layerCallPauses is turned off during this step
    this.inputLayerNames.forEach(name => {
      const inputLayer = this.modelLayersMap.get(name);
      inputLayer.call(this.inputTensorsMap.get(name));
      inputLayer.hasOutput = true;
      inputLayer.visited = true;
    });
    this._traverseDAG(this.inputLayerNames);

    // reset hasOutput and visited flags in all layers
    this.finishedLayerNames = [];
    this.modelLayersMap.forEach(layer => {
      layer.hasOutput = false;
      layer.visited = false;
    });

    return true;
  }

  /**
   * Makes data FS request (node only)
   *
   * @param {string} type - type of requested data, one of `model`, `weights`, or `metadata`.
   * @returns {Promise}
   */
  async _dataRequestFS(type) {
    const readFile = _bluebird2.default.promisify(require('fs').readFile);
    const filetype = this.filetypes[type];
    const encoding = filetype === 'json' ? 'utf8' : undefined;

    try {
      const data = await readFile(this.filepaths[type], encoding);
      if (filetype === 'json') {
        this.data[type] = JSON.parse(data);
      } else if (filetype === 'arraybuffer') {
        this.data[type] = data.buffer;
      } else {
        throw new Error(`[Model] Invalid file type: ${filetype}`);
      }
    } catch (err) {
      throw err;
    }
    this.dataRequestProgress[type] = 100;
  }

  /**
   * Makes data HTTP request (browser or node)
   *
   * @param {string} type - type of requested data, one of `model`, `weights`, or `metadata`.
   * @param {Object} [headers] - any headers to be passed along with request
   * @returns {Promise}
   */
  async _dataRequestHTTP(type, headers = {}) {
    try {
      const res = await _axios2.default.get(this.filepaths[type], {
        responseType: this.filetypes[type],
        headers,
        onDownloadProgress: e => {
          if (e.lengthComputable) {
            const percentComplete = Math.round(100 * e.loaded / e.total);
            this.dataRequestProgress[type] = percentComplete;
          }
        },
        cancelToken: axiosSource.token
      });
      this.data[type] = res.data;
    } catch (err) {
      if (_axios2.default.isCancel(err)) {
        console.log('Data request canceled', err.message);
      } else {
        throw err;
      }
    }
    this.dataRequestProgress[type] = 100;
  }

  /**
   * Loading progress calculated from all the data requests combined.
   * @returns {number} progress
   */
  getLoadingProgress() {
    const progressValues = (0, _values3.default)(this.dataRequestProgress);
    return Math.round((0, _sum3.default)(progressValues) / progressValues.length);
  }

  /**
   * Toggle GPU mode on/off
   * Iterate through all layers and set `gpu` attribute
   * @param {boolean} mode - on/off
   */
  toggleGPU(mode) {
    if (typeof mode === 'undefined') {
      this.gpu = !this.gpu;
    } else {
      this.gpu = mode;
    }
    this.modelLayersMap.forEach(layer => {
      layer.toggleGPU(this.gpu);
    });
    this.resetInputTensors();
  }

  /**
   * Resets input tensors
   */
  resetInputTensors() {
    this.inputLayerNames.forEach(name => {
      const inputLayer = this.modelLayersMap.get(name);
      this.inputTensorsMap.set(name, new _Tensor2.default([], inputLayer.shape));
    });
  }

  /**
   * Builds directed acyclic graph of model layers
   *
   * Every layer in the model defines inbound and outbound nodes. For Keras models of class Sequential, we still convert
   * the list into DAG format for straightforward interoperability with graph models (however, we must first create an
   * Input layer as the initial layer. For class Model, the DAG is constructed from the configuration inbound and
   * outbound nodes. Note that Models can have layers be entire Sequential branches.
   */
  _buildDAG() {
    const modelClass = this.data.model.class_name;

    let modelConfig = [];
    if (modelClass === 'Sequential') {
      modelConfig = this.data.model.config;
    } else if (modelClass === 'Model') {
      modelConfig = this.data.model.config.layers;
    }

    if (!(Array.isArray(modelConfig) && modelConfig.length)) {
      throw new Error('[Model] Model configuration does not contain any layers.');
    }

    modelConfig.forEach((layerDef, index) => {
      const layerClass = layerDef.class_name;
      const layerConfig = layerDef.config;

      if (modelClass === 'Model' && layerClass === 'Sequential') {
        // when layer is a Sequential branch in a Model
        layerConfig.forEach((branchLayerDef, branchIndex) => {
          const branchLayerClass = branchLayerDef.class_name;
          const branchLayerConfig = branchLayerDef.config;

          const branchInboundLayerNames = branchIndex === 0 ? layerDef.inbound_nodes[0].map(node => node[0]) : [layerConfig[branchIndex - 1].config.name];

          this._createLayer(branchLayerClass, branchLayerConfig, branchInboundLayerNames);
        });
      } else if (!(layerClass in layers)) {
        throw new Error(`[Model] Layer ${layerClass} specified in model configuration is not implemented!`);
      } else {
        // create InputLayer node for Sequential class (which is not explicitly defined in config)
        // create input tensor for InputLayer specified in Model class (layer itself created later)
        if (modelClass === 'Sequential' && index === 0) {
          const inputName = 'input';
          const inputShape = layerConfig.batch_input_shape.slice(1);
          const layer = new layers.InputLayer({ name: inputName, shape: inputShape, gpu: this.gpu });
          this.modelLayersMap.set(inputName, layer);
          this.inputTensorsMap.set(inputName, new _Tensor2.default([], inputShape));
          this.inputLayerNames.push(inputName);
        } else if (modelClass === 'Model' && layerClass === 'InputLayer') {
          const inputShape = layerConfig.batch_input_shape.slice(1);
          this.inputTensorsMap.set(layerConfig.name, new _Tensor2.default([], inputShape));
          this.inputLayerNames.push(layerConfig.name);
        }

        let inboundLayerNames = [];
        if (modelClass === 'Sequential') {
          if (index === 0) {
            inboundLayerNames = ['input'];
          } else {
            inboundLayerNames = [modelConfig[index - 1].config.name];
          }
        } else if (modelClass === 'Model') {
          const inboundNodes = layerDef.inbound_nodes;
          if (inboundNodes && inboundNodes.length) {
            inboundLayerNames = inboundNodes[0].map(node => node[0]);
          }
        }
        this._createLayer(layerClass, layerConfig, inboundLayerNames);
      }
    });

    this.modelLayersMap.forEach(layer => {
      if (layer.outbound.length === 0) {
        this.outputLayerNames.push(layer.name);
      }
    });

    this.inputLayerNames.sort();
    this.outputLayerNames.sort();
  }

  /**
   * Create single layer
   *
   * @param {Object} layerClass
   * @param {Object} layerConfig
   * @param {string[]} inboundLayerNames
   */
  _createLayer(layerClass, layerConfig, inboundLayerNames) {
    const layer = new layers[layerClass](Object.assign({}, layerConfig, { gpu: this.gpu }));

    // layer weights
    let weightNames = [];
    if (layerClass === 'Bidirectional') {
      const forwardWeightNames = layer.forwardLayer.params.map(param => `${layerConfig.name}/forward_${layerConfig.layer.config.name}/${param}`);
      const backwardWeightNames = layer.backwardLayer.params.map(param => `${layerConfig.name}/backward_${layerConfig.layer.config.name}/${param}`);
      weightNames = forwardWeightNames.concat(backwardWeightNames);
    } else if (layerClass === 'TimeDistributed') {
      weightNames = layer.layer.params.map(param => `${layerConfig.name}/${param}`);
    } else {
      weightNames = layer.params.map(param => `${layerConfig.name}/${param}`);
    }
    if (weightNames && weightNames.length) {
      const weights = weightNames.map(weightName => {
        const paramMetadata = (0, _find3.default)(this.data.metadata, meta => {
          const weightRE = new RegExp(`^${weightName}`);
          return weightRE.test(meta.weight_name);
        });
        if (!paramMetadata) {
          throw new Error(`[Model] error loading weights.`);
        }
        const { offset, length, shape } = paramMetadata;
        return new _Tensor2.default(new Float32Array(this.data.weights, offset, length), shape);
      });
      layer.setWeights(weights);
    }

    this.modelLayersMap.set(layerConfig.name, layer);

    inboundLayerNames.forEach(layerName => {
      this.modelLayersMap.get(layerConfig.name).inbound.push(layerName);
      this.modelLayersMap.get(layerName).outbound.push(layerConfig.name);
    });
  }

  /**
   * Async function for recursively traversing the DAG
   * Graph object is stored in `this.modelDAG`, keyed by layer name.
   * Layers are retrieved from Map object `this.modelLayersMap`.
   *
   * @param {string[]} nodes - array of layer names
   * @returns {Promise}
   */
  _traverseDAG(nodes) {
    if (nodes.length === 0) {
      // Stopping criterion:
      // an output node will have 0 outbound nodes.
      return true;
    } else if (nodes.length === 1) {
      // Where computational logic lives for a given layer node
      // - Makes sure outputs are available from inbound layer nodes
      // - Keeps async function going until outputs are available from inbound layer nodes
      //   (important for merge layer nodes where multiple inbound nodes may complete asynchronously)
      // - Runs computation for current layer node: .call()
      // - Starts new async function for outbound nodes
      const node = nodes[0];
      const currentLayer = this.modelLayersMap.get(node);

      if (currentLayer.layerClass === 'InputLayer') {
        this.finishedLayerNames.push(this.modelLayersMap.get(node).name);
      } else {
        const currentLayer = this.modelLayersMap.get(node);
        if (currentLayer.visited) {
          return false;
        }

        const inboundLayers = currentLayer.inbound.map(n => this.modelLayersMap.get(n));
        if (!(0, _every3.default)((0, _map3.default)(inboundLayers, 'hasOutput'))) {
          return false;
        }

        if (currentLayer.isMergeLayer) {
          currentLayer.call((0, _map3.default)(inboundLayers, 'output'));
        } else {
          currentLayer.call(inboundLayers[0].output);
        }

        currentLayer.hasOutput = true;
        currentLayer.visited = true;
        this.finishedLayerNames.push(currentLayer.name);
      }

      this._traverseDAG(currentLayer.outbound);
    } else {
      nodes.forEach(node => this._traverseDAG([node]));
    }
  }

  /**
   * Transfer intermediate outputs if specified, only in GPU mode and if transferLayerOutputs is set to true
   */
  _maybeTransferIntermediateOutputs() {
    if (this.gpu && this.transferLayerOutputs) {
      this.modelLayersMap.forEach(layer => {
        if (layer.output && layer.output.glTexture) {
          _WebGL.webgl2.bindOutputTexture(layer.output.glTexture, layer.output.glTextureShape);
          layer.output.transferFromGLTexture();
          if (layer.output.is2DReshaped) {
            layer.output.reshapeFrom2D();
          }
        }
      });
    }
  }

  /**
   * Load data to input layer nodes
   *
   * @param {Object} inputData - object where the keys are the named inputs of the model,
   * and values the TypedArray numeric data
   */
  loadData(inputData) {
    this.inputLayerNames.forEach(name => {
      const inputLayer = this.modelLayersMap.get(name);
      this.inputTensorsMap.get(name).replaceTensorData(inputData[name]);
      inputLayer.call(this.inputTensorsMap.get(name));
      inputLayer.hasOutput = true;
      inputLayer.visited = true;
    });
  }

  /**
   * Predict
   *
   * @param {Object} inputData - object where the keys are the named inputs of the model,
   * and values the TypedArray numeric data
   * @returns {Promise} - outputData object where the keys are the named outputs of the model,
   * and values the TypedArray numeric data
   */
  predict(inputData) {
    this.isRunning = true;

    if (!(0, _isEqual3.default)((0, _keys3.default)(inputData).sort(), this.inputLayerNames)) {
      this.isRunning = false;
      throw new Error(`[Model] predict() must take an object where the keys are the named inputs of the model: ${this.inputLayerNames}.`);
    }
    if (!(0, _every3.default)(this.inputLayerNames, name => inputData[name] instanceof Float32Array)) {
      this.isRunning = false;
      throw new Error('[Model] predict() must take an object where the values are the flattened data as Float32Array.');
    }

    // reset hasOutput and visited flags in all layers
    this.finishedLayerNames = [];
    this.modelLayersMap.forEach(layer => {
      layer.hasOutput = false;
      layer.visited = false;
    });

    // load data to input tensors
    this.loadData(inputData);

    // start traversing DAG at inputs
    this._traverseDAG(this.inputLayerNames);

    // transfer intermediate outputs if specified
    this._maybeTransferIntermediateOutputs();

    // outputs are layers with no outbound nodes
    const modelClass = this.data.model.class_name;
    const outputData = {};
    if (modelClass === 'Sequential') {
      const outputLayer = this.modelLayersMap.get(this.outputLayerNames[0]);
      outputData['output'] = outputLayer.output.tensor.data;
    } else if (modelClass === 'Model') {
      this.outputLayerNames.forEach(layerName => {
        const outputLayer = this.modelLayersMap.get(layerName);
        outputData[layerName] = outputLayer.output.tensor.data;
      });
    }

    this.isRunning = false;
    return outputData;
  }
}
exports.default = Model;