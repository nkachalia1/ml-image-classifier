/**
 * ==========================================================================
 * NEURALSIGHT ML BACKEND ENGINE (classifier.js)
 * Manages TensorFlow.js initialization, MobileNet, and KNN Transfer Learning
 * ==========================================================================
 */

const ClassifierEngine = {
    // Models
    mobileNetModel: null,
    knnClassifierInstance: null,
    gradCAMModel: null,
    gradCAMSubModel: null,
    gradCAMLayerName: null,
    
    activeModelId: 'mobilenet', // 'mobilenet', 'resnet', 'cocossd'
    resnetModel: null,
    cocoSsdModel: null,
    cocoSsdLoadPromise: null,
    imagenetLabels: null,
    personAssistUnavailable: false,
    personAssistCache: {
        timestamp: -Infinity,
        element: null,
        detections: []
    },
    personAssistIntervalMs: 650,
    personAssistMinScore: 0.42,
    
    // Status
    isLoaded: false,
    backend: 'cpu',
    numClasses: 0,
    classNamesMap: {}, // Maps numeric index to string labels
    minCustomClassesForPrediction: 2,
    customUnknownSimilarityFloor: 0.985,
    customFullConfidenceSimilarity: 0.995,
    customStandardActivationFloor: 0.988,
    customStandardActivationMargin: 0.002,
    customStandardBaselineAlpha: 0.04,
    customStandardState: {},
    
    // UI Event listeners hooks
    onStatusChange: null,
    onModelLoad: null,

    /**
     * Initializes TensorFlow.js and loads the core MobileNet model
     */
    async init() {
        this.updateStatus('tf-status', 'loading', 'TF.js: Loading...');
        
        try {
            // Wait for TFJS to be ready
            await tf.ready();
            
            // Load ImageNet labels for ResNet mapping
            try {
                const response = await fetch('https://cdn.jsdelivr.net/gh/anishathalye/imagenet-simple-labels@master/imagenet-simple-labels.json');
                this.imagenetLabels = await response.json();
                console.log('[NeuralSight] ImageNet labels loaded successfully from CDN.');
            } catch (err) {
                console.error('[NeuralSight] Failed to load ImageNet labels from CDN:', err);
            }
            
            // Set optimal backend (WebGL preferred, WASM fallback, CPU last)
            this.backend = tf.getBackend();
            console.log(`[NeuralSight] TensorFlow.js initialized on backend: ${this.backend}`);
            this.updateStatus('tf-status', 'online', `TF.js: ${this.backend.toUpperCase()}`);
            
            // Initialize KNN Classifier
            this.knnClassifierInstance = knnClassifier.create();
            console.log('[NeuralSight] KNN Classifier instance created.');
            
            // Load MobileNet
            this.updateStatus('model-status', 'loading', 'MobileNet: Loading...');
            
            // Loading MobileNet v2 with 1.0 alpha multiplier and 224px input size
            this.mobileNetModel = await mobilenet.load({
                version: 2,
                alpha: 1.0
            });
            
            // Initialize Grad-CAM sub-models asynchronously
            await this.initGradCAM();
            
            this.isLoaded = true;
            console.log('[NeuralSight] MobileNet v2 model loaded successfully.');
            this.updateStatus('model-status', 'online', this.getMobileNetStatusLabel());
            
            if (this.onModelLoad) {
                this.onModelLoad();
            }

            // Start the human detector in the background so MobileNet can avoid
            // ImageNet-only false positives such as "shower cap" for webcam users.
            this.ensureCocoSsdModel({ silent: true }).catch(err => {
                console.warn('[NeuralSight] Person assist unavailable; continuing with MobileNet only.', err);
            });
        } catch (error) {
            console.error('[NeuralSight] Error during ML backend initialization:', error);
            this.updateStatus('tf-status', 'offline', 'TF.js: Init Failed');
            this.updateStatus('model-status', 'offline', 'MobileNet: Load Failed');
        }
    },

    /**
     * Set updates for status badges in the UI header
     */
    updateStatus(badgeId, state, labelText) {
        if (this.onStatusChange) {
            this.onStatusChange(badgeId, state, labelText);
        }
    },

    /**
     * Load a model dynamically on demand
     * @param {string} modelId
     */
    async loadModel(modelId) {
        if (modelId === this.activeModelId) return;
        
        let targetLabel = '';
        if (modelId === 'mobilenet') targetLabel = 'MobileNet';
        else if (modelId === 'resnet') targetLabel = 'ResNet';
        else if (modelId === 'cocossd') targetLabel = 'COCO-SSD';
        
        this.updateStatus('model-status', 'loading', `${targetLabel}: Loading...`);
        console.log(`[NeuralSight] Loading model dynamically: ${modelId}`);
        
        try {
            if (modelId === 'mobilenet') {
                // Already loaded during startup init()
                this.activeModelId = 'mobilenet';
                this.updateStatus('model-status', 'online', this.getMobileNetStatusLabel());
            } else if (modelId === 'resnet') {
                if (!this.resnetModel) {
                    this.resnetModel = await tf.loadLayersModel('https://raw.githubusercontent.com/paulsp94/tfjs_resnet_imagenet/master/ResNet50/model.json');
                }
                this.activeModelId = 'resnet';
                this.updateStatus('model-status', 'online', 'ResNet: Ready');
            } else if (modelId === 'cocossd') {
                await this.ensureCocoSsdModel({ silent: false });
                this.activeModelId = 'cocossd';
                this.updateStatus('model-status', 'online', 'COCO-SSD: Ready');
            }
            console.log(`[NeuralSight] Dynamic model ${modelId} successfully loaded.`);
        } catch (error) {
            console.error(`[NeuralSight] Failed to dynamically load model ${modelId}:`, error);
            this.updateStatus('model-status', 'offline', `${targetLabel}: Load Failed`);
            throw error;
        }
    },

    getMobileNetStatusLabel() {
        return this.cocoSsdModel ? 'MobileNet + Person: Ready' : 'MobileNet: Ready';
    },

    async ensureCocoSsdModel({ silent = false } = {}) {
        if (this.cocoSsdModel) {
            return this.cocoSsdModel;
        }

        if (this.personAssistUnavailable) {
            return null;
        }

        if (!window.cocoSsd) {
            this.personAssistUnavailable = true;
            return null;
        }

        if (!this.cocoSsdLoadPromise) {
            if (!silent && this.activeModelId === 'mobilenet') {
                this.updateStatus('model-status', 'loading', 'MobileNet: Loading person assist...');
            }

            this.cocoSsdLoadPromise = cocoSsd.load()
                .then(model => {
                    this.cocoSsdModel = model;
                    console.log('[NeuralSight] COCO-SSD person assist loaded for MobileNet mode.');

                    if (this.activeModelId === 'mobilenet') {
                        this.updateStatus('model-status', 'online', this.getMobileNetStatusLabel());
                    }

                    return model;
                })
                .catch(error => {
                    this.personAssistUnavailable = true;
                    console.error('[NeuralSight] Failed to load COCO-SSD person assist:', error);

                    if (this.activeModelId === 'mobilenet') {
                        this.updateStatus('model-status', 'online', 'MobileNet: Ready');
                    }

                    return null;
                })
                .finally(() => {
                    this.cocoSsdLoadPromise = null;
                });
        }

        return await this.cocoSsdLoadPromise;
    },

    /**
     * Run inference using standard ResNet50
     */
    async predictResNet(element) {
        if (!this.resnetModel) {
            throw new Error('ResNet50 model is not loaded yet');
        }
        
        const preprocessedCanvas = this.preprocessToCanvas(element);
        
        const predictions = tf.tidy(() => {
            const tensor = tf.browser.fromPixels(preprocessedCanvas);
            
            // Standard Keras ImageNet preprocessing mean values (BGR subtraction)
            const offset = tf.tensor1d([123.68, 116.779, 103.939]);
            const preprocessed = tensor.toFloat().sub(offset).expandDims(0);
            
            let logits = this.resnetModel.predict(preprocessed);
            if (Array.isArray(logits) && logits.length > 0) {
                logits = logits[0];
            } else if (logits && typeof logits === 'object' && !(logits instanceof tf.Tensor)) {
                const keys = Object.keys(logits);
                if (keys.length > 0) {
                    logits = logits[keys[0]];
                }
            }
            const probabilities = tf.softmax(logits);
            return probabilities.squeeze().dataSync();
        });
        
        const predictionsWithIndices = Array.from(predictions).map((prob, index) => ({
            probability: prob,
            index: index
        }));
        
        predictionsWithIndices.sort((a, b) => b.probability - a.probability);
        
        const top3 = predictionsWithIndices.slice(0, 3).map(pred => {
            const className = this.imagenetLabels ? this.imagenetLabels[pred.index] : `Class ${pred.index}`;
            return {
                className: className,
                probability: pred.probability
            };
        });
        
        return top3;
    },



    /**
     * Run inference using COCO-SSD
     */
    async predictCocoSsd(element) {
        await this.ensureCocoSsdModel({ silent: false });

        if (!this.cocoSsdModel) {
            throw new Error('COCO-SSD model is not loaded yet');
        }
        return await this.cocoSsdModel.detect(element);
    },

    /**
     * Predict labels on an image or video element using standard MobileNet
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} element 
     * @returns {Promise<Array>} List of predictions with className and probability
     */
    async predictStandard(element) {
        if (!this.isLoaded) {
            throw new Error('Classifier engine is not initialized');
        }
        
        if (this.activeModelId === 'mobilenet') {
            if (!this.mobileNetModel) throw new Error('MobileNet model is not loaded yet');
            const mobileNetPredictions = await this.mobileNetModel.classify(element, 3);
            return await this.applyPersonAssist(element, mobileNetPredictions);
        } else if (this.activeModelId === 'resnet') {
            return await this.predictResNet(element);
        } else if (this.activeModelId === 'cocossd') {
            const detections = await this.predictCocoSsd(element);
            return detections.map(pred => ({
                className: pred.class,
                probability: pred.score,
                bbox: pred.bbox
            }));
        }
        
        throw new Error(`Unsupported active model: ${this.activeModelId}`);
    },

    async applyPersonAssist(element, mobileNetPredictions) {
        const detections = await this.getCachedObjectDetections(element);
        const personDetections = detections
            .filter(pred => pred.class === 'person' && pred.score >= this.personAssistMinScore)
            .sort((a, b) => b.score - a.score);

        if (personDetections.length === 0) {
            return mobileNetPredictions;
        }

        const topPerson = personDetections[0];
        const otherObjects = detections
            .filter(pred => pred.class !== 'person' && pred.score >= 0.5)
            .sort((a, b) => b.score - a.score);

        const detectorPredictions = [
            this.toDetectorPrediction(topPerson),
            ...otherObjects.map(pred => this.toDetectorPrediction(pred))
        ];

        return detectorPredictions.slice(0, 3);
    },

    async getCachedObjectDetections(element) {
        const now = performance.now();
        const cacheAge = now - this.personAssistCache.timestamp;

        if (this.personAssistCache.element === element && cacheAge < this.personAssistIntervalMs) {
            return this.personAssistCache.detections;
        }

        const model = await this.ensureCocoSsdModel({ silent: true });
        if (!model) {
            return [];
        }

        try {
            const detections = await model.detect(element);
            this.personAssistCache = {
                timestamp: performance.now(),
                element,
                detections
            };
            return detections;
        } catch (error) {
            console.warn('[NeuralSight] Person assist detection skipped for this frame:', error);
            return this.personAssistCache.detections || [];
        }
    },

    toDetectorPrediction(prediction) {
        return {
            className: prediction.class,
            probability: prediction.score,
            bbox: prediction.bbox,
            isDetectorAssist: true
        };
    },

    // Offscreen Canvas for persistent, clean pre-processing resizing
    offscreenCanvas: null,
    offscreenCtx: null,

    getOffscreenCanvas() {
        if (!this.offscreenCanvas) {
            this.offscreenCanvas = document.createElement('canvas');
            this.offscreenCanvas.width = 224;
            this.offscreenCanvas.height = 224;
            this.offscreenCtx = this.offscreenCanvas.getContext('2d');
        }
        return this.offscreenCanvas;
    },

    preprocessToCanvas(element) {
        const canvas = this.getOffscreenCanvas();
        const ctx = this.offscreenCtx;
        ctx.clearRect(0, 0, 224, 224);
        
        let srcWidth = element.width || element.videoWidth || 224;
        let srcHeight = element.height || element.videoHeight || 224;
        
        if (element.naturalWidth) srcWidth = element.naturalWidth;
        if (element.naturalHeight) srcHeight = element.naturalHeight;
        
        let minDim = Math.min(srcWidth, srcHeight);
        if (minDim <= 0) minDim = 224;
        
        // Center crop image calculation to avoid aspect ratio squishing
        const sx = (srcWidth - minDim) / 2;
        const sy = (srcHeight - minDim) / 2;
        
        ctx.drawImage(element, sx, sy, minDim, minDim, 0, 0, 224, 224);
        return canvas;
    },

    /**
     * Extract bottleneck activation tensors from MobileNet (embedding vector)
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} element 
     * @returns {tf.Tensor} Activation tensor of shape [1, 1024]
     */
    getActivation(element) {
        if (!this.isLoaded || !this.mobileNetModel) {
            throw new Error('Classifier model not loaded');
        }
        
        // Preprocess any dimension element to a center-cropped 224x224 canvas
        const preprocessedCanvas = this.preprocessToCanvas(element);
        
        // Wrap execution in tf.tidy to automatically clean up intermediate tensors in WebGL memory
        return tf.tidy(() => {
            const tensor = tf.browser.fromPixels(preprocessedCanvas);
            
            // Normalize tensor values to be between -1 and 1 (standard for MobileNet)
            const normalized = tensor.toFloat().sub(tf.scalar(127.5)).div(tf.scalar(127.5));
            
            // Get intermediate activation logits (the pre-classification features layer)
            return this.mobileNetModel.infer(normalized, true);
        });
    },

    addCustomExample(element, classId, className) {
        if (!this.knnClassifierInstance) {
            throw new Error('KNN Classifier is not initialized');
        }
        
        // Map class ID to its display name
        this.classNamesMap[classId] = className;
        delete this.customStandardState[classId];
        
        // 1. Add Original Sample Vector
        const activation = this.getActivation(element);
        this.knnClassifierInstance.addExample(activation, classId);
        activation.dispose();
        
        // 2. Perform Mathematical Data Augmentation in-browser
        // Spawns 4 augmented vectors (Flip, Rotate Left, Rotate Right, Brightness Shift)
        // This expands training coverage 5x to create rotation, perspective and lighting invariance!
        this.augmentAndTrain(element, classId);
        
        this.updateNumClasses();
        console.log(`[NeuralSight] Recorded sample and 6 augmented vectors for class '${className}' (ID: ${classId})`);
    },

    /**
     * Creates 6 visual augmentations on offscreen canvas and registers them into the KNN classifier dataset
     */
    augmentAndTrain(element, classId) {
        const canvas = this.getOffscreenCanvas();
        const ctx = this.offscreenCtx;
        
        let srcWidth = element.width || element.videoWidth || 224;
        let srcHeight = element.height || element.videoHeight || 224;
        if (element.naturalWidth) srcWidth = element.naturalWidth;
        if (element.naturalHeight) srcHeight = element.naturalHeight;
        
        let minDim = Math.min(srcWidth, srcHeight);
        if (minDim <= 0) minDim = 224;
        
        const sx = (srcWidth - minDim) / 2;
        const sy = (srcHeight - minDim) / 2;
        
        // Sub-routine to convolve the offscreen canvas frame and feed embedding to model
        const registerAugmented = () => {
            tf.tidy(() => {
                const tensor = tf.browser.fromPixels(canvas);
                const normalized = tensor.toFloat().sub(tf.scalar(127.5)).div(tf.scalar(127.5));
                const activationTensor = this.mobileNetModel.infer(normalized, true);
                this.knnClassifierInstance.addExample(activationTensor, classId);
            });
        };
        
        // Augmentation 1: Horizontal Flip (Mirror perspective invariance)
        ctx.clearRect(0, 0, 224, 224);
        ctx.save();
        ctx.translate(224, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(element, sx, sy, minDim, minDim, 0, 0, 224, 224);
        ctx.restore();
        registerAugmented();
        
        // Augmentation 2: Rotate Left -6 degrees (Tilt perspective invariance)
        ctx.clearRect(0, 0, 224, 224);
        ctx.save();
        ctx.translate(112, 112);
        ctx.rotate(-6 * Math.PI / 180);
        ctx.drawImage(element, sx, sy, minDim, minDim, -112, -112, 224, 224);
        ctx.restore();
        registerAugmented();
        
        // Augmentation 3: Rotate Right +6 degrees (Tilt perspective invariance)
        ctx.clearRect(0, 0, 224, 224);
        ctx.save();
        ctx.translate(112, 112);
        ctx.rotate(6 * Math.PI / 180);
        ctx.drawImage(element, sx, sy, minDim, minDim, -112, -112, 224, 224);
        ctx.restore();
        registerAugmented();
        
        // Augmentation 4: Brightness Shift 1.15x (Room lighting variation invariance)
        ctx.clearRect(0, 0, 224, 224);
        ctx.save();
        ctx.filter = 'brightness(1.15) contrast(1.05)';
        ctx.drawImage(element, sx, sy, minDim, minDim, 0, 0, 224, 224);
        ctx.restore();
        registerAugmented();
        
        // Augmentation 5: Zoom/Crop 1.15x (Scale/distance invariance)
        ctx.clearRect(0, 0, 224, 224);
        ctx.save();
        const zoomDim = minDim * 0.85;
        const zsx = sx + (minDim - zoomDim) / 2;
        const zsy = sy + (minDim - zoomDim) / 2;
        ctx.drawImage(element, zsx, zsy, zoomDim, zoomDim, 0, 0, 224, 224);
        ctx.restore();
        registerAugmented();
        
        // Augmentation 6: Dimmer Lighting (Low-light variation invariance)
        ctx.clearRect(0, 0, 224, 224);
        ctx.save();
        ctx.filter = 'brightness(0.8) contrast(0.9)';
        ctx.drawImage(element, sx, sy, minDim, minDim, 0, 0, 224, 224);
        ctx.restore();
        registerAugmented();
    },

    /**
     * Classifies a target image feed using the custom transfer-learned model
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} element 
     * @returns {Promise<Object>} Predicted class index, name, and full list of confidences
     */
    async predictCustom(element) {
        return await this.predictCustomWithOptions(element, {
            minClasses: this.minCustomClassesForPrediction
        });
    },

    async predictCustomWithOptions(element, options = {}) {
        const minClasses = options.minClasses || this.minCustomClassesForPrediction;

        if (!this.knnClassifierInstance || this.knnClassifierInstance.getNumClasses() < minClasses) {
            return null;
        }
        
        const activation = this.getActivation(element);
        
        try {
            const dataset = this.knnClassifierInstance.getClassifierDataset();
            const totalExamples = Object.values(dataset).reduce((sum, t) => sum + t.shape[0], 0);
            const k = Math.min(5, Math.max(1, Math.floor(totalExamples / 3)));

            // Predict label using k-Nearest Neighbors clustering
            const result = await this.knnClassifierInstance.predictClass(activation, k);
            
            // Calculate cosine similarity between query activation and trained datasets
            const queryVal = activation.dataSync();
            const vectorLength = queryVal.length; // Dynamically grab model flat features count
            
            let queryMean = 0;
            for (let i = 0; i < vectorLength; i++) {
                queryMean += queryVal[i];
            }
            queryMean /= vectorLength;
            
            const queryZeroMean = new Float32Array(vectorLength);
            let queryNorm = 0;
            for (let i = 0; i < vectorLength; i++) {
                queryZeroMean[i] = queryVal[i] - queryMean;
                queryNorm += queryZeroMean[i] * queryZeroMean[i];
            }
            queryNorm = Math.sqrt(queryNorm);
            
            const predictionsList = [];
            const confidences = result.confidences; // Key: classId, Value: probability [0-1]
            
            for (const id in confidences) {
                let maxSimilarity = 0;
                const classTensor = dataset[id];
                
                if (classTensor) {
                    const classVal = classTensor.dataSync();
                    const numExamples = classTensor.shape[0];
                    
                    // Loop over each training sample's feature slice
                    for (let ex = 0; ex < numExamples; ex++) {
                        const offset = ex * vectorLength;
                        
                        let exMean = 0;
                        for (let i = 0; i < vectorLength; i++) {
                            exMean += classVal[offset + i];
                        }
                        exMean /= vectorLength;
                        
                        let dotProduct = 0;
                        let exNorm = 0;
                        for (let i = 0; i < vectorLength; i++) {
                            const valZeroMean = classVal[offset + i] - exMean;
                            dotProduct += queryZeroMean[i] * valZeroMean;
                            exNorm += valZeroMean * valZeroMean;
                        }
                        
                        exNorm = Math.sqrt(exNorm);
                        const similarity = (queryNorm && exNorm) ? (dotProduct / (queryNorm * exNorm)) : 0;
                        
                        if (similarity > maxSimilarity) {
                            maxSimilarity = similarity;
                        }
                    }
                }
                
                // Soft thresholding similarity scale:
                // Lower similarities are treated as unrelated to prevent a trained
                // class from sticking to unrelated webcam frames.
                let probability = confidences[id];
                if (maxSimilarity < this.customUnknownSimilarityFloor) {
                    probability = 0.0;
                } else if (maxSimilarity < this.customFullConfidenceSimilarity) {
                    // Smooth linear interpolation multiplier
                    const scale = (maxSimilarity - this.customUnknownSimilarityFloor) / (this.customFullConfidenceSimilarity - this.customUnknownSimilarityFloor);
                    probability = confidences[id] * scale;
                }
                
                predictionsList.push({
                    classId: parseInt(id),
                    className: this.classNamesMap[id] || `Class ${id}`,
                    probability: probability,
                    similarity: maxSimilarity
                });
            }
            
            // Sort by highest probability
            predictionsList.sort((a, b) => b.probability - a.probability);
            
            return {
                label: predictionsList[0].probability > 0 ? predictionsList[0].className : 'Unknown / Generic',
                classIndex: predictionsList[0].probability > 0 ? predictionsList[0].classId : -1,
                predictions: predictionsList
            };
        } finally {
            // Dispose activation tensor to prevent memory leakage
            activation.dispose();
        }
    },

    async predictCustomForStandard(element) {
        const result = await this.predictCustomWithOptions(element, { minClasses: 1 });
        if (!result || !result.predictions) return [];

        const proposals = [];
        const usesLiveBaseline = element.tagName === 'VIDEO';

        result.predictions.forEach(pred => {
            const similarity = pred.similarity || 0;

            if (!usesLiveBaseline) {
                if (similarity >= this.customUnknownSimilarityFloor && pred.probability >= 0.75) {
                    proposals.push({
                        className: pred.className,
                        probability: Math.min(0.98, Math.max(0.76, pred.probability)),
                        isCustom: true
                    });
                }
                return;
            }

            const state = this.customStandardState[pred.classId] || {
                baseline: null,
                lastSimilarity: 0
            };

            if (state.baseline === null || !Number.isFinite(state.baseline)) {
                state.baseline = similarity;
                state.lastSimilarity = similarity;
                this.customStandardState[pred.classId] = state;
                return;
            }

            const lift = similarity - state.baseline;
            const isActiveMatch = similarity >= this.customStandardActivationFloor &&
                                  lift >= this.customStandardActivationMargin;

            if (isActiveMatch) {
                const confidence = Math.min(0.98, Math.max(0.76, 0.76 + lift * 45));
                proposals.push({
                    className: pred.className,
                    probability: confidence,
                    isCustom: true
                });

                // Let the baseline follow slowly so a held object does not instantly
                // become the new background, but long-running scene changes recover.
                const cappedSimilarity = Math.min(similarity, state.baseline + this.customStandardActivationMargin);
                state.baseline = state.baseline * (1 - this.customStandardBaselineAlpha) + cappedSimilarity * this.customStandardBaselineAlpha;
            } else {
                state.baseline = state.baseline * (1 - this.customStandardBaselineAlpha) + similarity * this.customStandardBaselineAlpha;
            }

            state.lastSimilarity = similarity;
            this.customStandardState[pred.classId] = state;
        });

        proposals.sort((a, b) => b.probability - a.probability);
        return proposals.slice(0, 2);
    },

    resetCustomStandardState() {
        this.customStandardState = {};
    },

    /**
     * Clear examples from a single custom class
     */
    clearClass(classId) {
        if (this.knnClassifierInstance) {
            this.knnClassifierInstance.clearClass(classId);
            delete this.classNamesMap[classId];
            delete this.customStandardState[classId];
            this.updateNumClasses();
            console.log(`[NeuralSight] Cleared all examples for class ID: ${classId}`);
        }
    },

    /**
     * Clear all examples from all classes in the KNN Classifier
     */
    resetCustomClassifier() {
        if (this.knnClassifierInstance) {
            this.knnClassifierInstance.clearAllClasses();
            this.classNamesMap = {};
            this.resetCustomStandardState();
            this.updateNumClasses();
            console.log('[NeuralSight] Reset custom transfer learning model.');
        }
    },

    /**
     * Updates the count of active classes
     */
    updateNumClasses() {
        if (this.knnClassifierInstance) {
            this.numClasses = this.knnClassifierInstance.getNumClasses();
        }
    },

    /**
     * Export the custom classifier dataset as a serializable JSON string
     * @returns {string} Serialized JSON object mapping class datasets
     */
    exportDataset() {
        if (!this.knnClassifierInstance || this.numClasses === 0) {
            return null;
        }
        
        // Grab dataset tensors
        const dataset = this.knnClassifierInstance.getClassifierDataset();
        const serializableDataset = {};
        
        // Convert tensors to regular JS arrays
        Object.keys(dataset).forEach(key => {
            const tensor = dataset[key];
            const values = tensor.dataSync();
            serializableDataset[key] = {
                shape: tensor.shape,
                values: Array.from(values)
            };
        });
        
        const packageData = {
            classNamesMap: this.classNamesMap,
            dataset: serializableDataset
        };
        
        return JSON.stringify(packageData);
    },

    /**
     * Import a previously serialized JSON dataset back into the KNN instance
     * @param {string} jsonString Serialized dataset packages
     */
    importDataset(jsonString) {
        if (!this.knnClassifierInstance) {
            throw new Error('KNN Classifier is not initialized');
        }
        
        const packageData = JSON.parse(jsonString);
        this.classNamesMap = packageData.classNamesMap;
        
        const serializableDataset = packageData.dataset;
        const dataset = {};
        
        Object.keys(serializableDataset).forEach(key => {
            const data = serializableDataset[key];
            dataset[key] = tf.tensor2d(data.values, data.shape);
        });
        
        this.knnClassifierInstance.setClassifierDataset(dataset);
        this.resetCustomStandardState();
        this.updateNumClasses();
        console.log('[NeuralSight] Custom dataset imported successfully.');
    },

    /**
     * Returns memory diagnostic data
     */
    getMemoryMetrics() {
        const memoryInfo = tf.memory();
        return {
            numTensors: memoryInfo.numTensors,
            numBytes: memoryInfo.numBytes,
            numMegabytes: (memoryInfo.numBytes / (1024 * 1024)).toFixed(2)
        };
    },

    /**
     * Initializes the Grad-CAM sub-models asynchronously by loading raw LayersModel
     */
    async initGradCAM() {
        if (this.gradCAMModel && this.gradCAMSubModel) {
            return; // already initialized
        }
        
        try {
            console.log('[NeuralSight] Loading MobileNet LayersModel for Grad-CAM activations...');
            const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
            const layerName = 'conv_pw_13_relu'; // standard MobileNet v1 final conv activation layer
            const convLayer = model.getLayer(layerName);
            
            if (convLayer) {
                this.gradCAMLayerName = layerName;
                
                // Get all layers coming AFTER 'conv_pw_13_relu'
                const layersAfter = [];
                let found = false;
                for (const layer of model.layers) {
                    if (found) {
                        layersAfter.push(layer);
                    }
                    if (layer.name === layerName) {
                        found = true;
                    }
                }
                
                // Create a sub-model starting from the output of the convolutional layer
                const convShape = convLayer.output.shape.slice(1);
                const symbolicInput = tf.input({ shape: convShape });
                let current = symbolicInput;
                for (const layer of layersAfter) {
                    current = layer.apply(current);
                }
                
                this.gradCAMSubModel = tf.model({ inputs: symbolicInput, outputs: current });
                
                // Also create a model that outputs the conv layer activations directly from input image
                this.gradCAMModel = tf.model({ inputs: model.inputs, outputs: convLayer.output });
                
                console.log(`[NeuralSight] Grad-CAM initialized successfully with layer: ${layerName}`);
            } else {
                console.error('[NeuralSight] Grad-CAM Layer not found:', layerName);
            }
        } catch (error) {
            console.error('[NeuralSight] Failed to initialize Grad-CAM:', error);
        }
    },

    /**
     * Extracts the class index of the highest probability prediction
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} element 
     */
    async getTopClassIndex(element) {
        if (!this.mobileNetModel || !this.mobileNetModel.model) {
            throw new Error('MobileNet model is not loaded');
        }
        
        const preprocessedCanvas = this.preprocessToCanvas(element);
        
        return tf.tidy(() => {
            const tensor = tf.browser.fromPixels(preprocessedCanvas);
            const normalized = tensor.toFloat().sub(tf.scalar(127.5)).div(tf.scalar(127.5));
            const inputTensor = normalized.expandDims(0);
            const logits = this.mobileNetModel.model.predict(inputTensor);
            const topIndexTensor = logits.argMax(1);
            return topIndexTensor.dataSync()[0];
        });
    },

    /**
     * Computes the Grad-CAM heatmap values for the target class index
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} element 
     * @param {number} targetClassIndex 
     * @returns {Promise<Float32Array>} 224x224 heatmap matrix normalized to [0, 1]
     */
    async computeGradCAM(element, targetClassIndex) {
        if (!this.gradCAMModel || !this.gradCAMSubModel) {
            await this.initGradCAM();
        }
        
        if (!this.gradCAMModel || !this.gradCAMSubModel) {
            throw new Error('Grad-CAM models are not initialized');
        }
        
        const preprocessedCanvas = this.preprocessToCanvas(element);
        
        return tf.tidy(() => {
            const tensor = tf.browser.fromPixels(preprocessedCanvas);
            const normalized = tensor.toFloat().sub(tf.scalar(127.5)).div(tf.scalar(127.5));
            const inputTensor = normalized.expandDims(0); // [1, 224, 224, 3]
            
            // 1. Get the intermediate feature activations
            const convActVal = this.gradCAMModel.predict(inputTensor); // [1, 7, 7, 1024]
            
            // 2. Define the score function w.r.t the intermediate activations
            const scoreFn = (convAct) => {
                const preds = this.gradCAMSubModel.predict(convAct); // [1, 1000]
                return preds.slice([0, targetClassIndex], [1, 1]).asScalar();
            };
            
            // 3. Compute gradients of score w.r.t feature activations
            const gradFn = tf.grad(scoreFn);
            const grads = gradFn(convActVal); // [1, 7, 7, 1024]
            
            // 4. Global Average Pooling over spatial dimensions (height & width)
            const weights = tf.mean(grads, [1, 2]); // [1, 1024]
            const weightsReshaped = weights.expandDims(1); // [1, 1, 1024]
            
            // 5. Multiply the activations by their importance weights and sum across channels
            const weightedSum = tf.mul(convActVal, weightsReshaped);
            const heatmap = tf.sum(weightedSum, 3).squeeze(); // [7, 7]
            
            // 6. Apply ReLU to keep only features that positively influence the prediction
            const reluHeatmap = tf.relu(heatmap);
            
            // 7. Normalize the heatmap values between 0 and 1
            const maxVal = reluHeatmap.max();
            const minVal = reluHeatmap.min();
            const denom = maxVal.sub(minVal).add(tf.scalar(1e-8));
            const normalizedHeatmap = reluHeatmap.sub(minVal).div(denom);
            
            // 8. Resize bilinear back to [224, 224] for overlay alignment
            const resizedHeatmap = tf.image.resizeBilinear(
                normalizedHeatmap.expandDims(-1).expandDims(0),
                [224, 224]
            ); // [1, 224, 224, 1]
            
            // Get Float32Array values
            return resizedHeatmap.squeeze().dataSync();
        });
    }
};

// Global Exposure
window.ClassifierEngine = ClassifierEngine;
