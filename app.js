/**
 * ==========================================================================
 * NEURALSIGHT SYSTEM COORDINATOR (app.js)
 * Orchestrates ML loops, UI updates, export/import actions and lifecycle
 * ==========================================================================
 */

const App = {
    // Prediction states
    isPredictingStandard: false,
    isPredictingCustom: false,
    
    /**
     * Initial startup hook
     */
    async init() {
        console.log('[NeuralSight] Bootstrapping application components...');
        
        // 1. Hook up status updates from Classifier Engine
        ClassifierEngine.onStatusChange = (badgeId, state, labelText) => {
            this.handleStatusUpdate(badgeId, state, labelText);
        };
        
        ClassifierEngine.onModelLoad = () => {
            this.handleModelLoaded();
        };

        // 2. Initialize UI panel bindings
        UIManager.init();
        
        // 3. Setup File Model IO triggers
        this.bindModelIO();
        
        // 4. Initialize ML Engine
        await ClassifierEngine.init();
        
        // 5. Start main high-performance render loop
        this.startRenderLoop();
    },

    /**
     * Updates header status badges visual state and contents
     */
    handleStatusUpdate(badgeId, state, labelText) {
        const badge = document.getElementById(badgeId);
        if (!badge) return;
        
        const dot = badge.querySelector('.status-dot');
        const label = badge.querySelector('.status-label');
        
        // Reset classes
        dot.className = 'status-dot';
        dot.classList.add(state);
        
        label.textContent = labelText;
    },

    /**
     * Fired when MobileNet is loaded and fully ready
     */
    handleModelLoaded() {
        // Automatically start the webcam as default source on launch
        UIManager.startWebcam();
        
        // Display backend details
        const backendText = ClassifierEngine.backend.toUpperCase();
        document.getElementById('backend-label').textContent = backendText;
        document.getElementById('inference-device').textContent = backendText;
    },

    /**
     * Setup JSON import and export triggers for KNN weights configuration
     */
    bindModelIO() {
        const exportBtn = document.getElementById('btn-export-model');
        const importInput = document.getElementById('model-import-input');
        const resetBtn = document.getElementById('btn-clear-classifier');
        
        // Export
        exportBtn.addEventListener('click', () => {
            const dataStr = ClassifierEngine.exportDataset();
            if (!dataStr) {
                alert('Dataset empty: Record custom samples first before exporting model.');
                return;
            }
            
            const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
            const exportFileDefaultName = `NeuralSight_Custom_Weights_${Date.now()}.json`;
            
            const linkElement = document.createElement('a');
            linkElement.setAttribute('href', dataUri);
            linkElement.setAttribute('download', exportFileDefaultName);
            linkElement.click();
            console.log('[NeuralSight] Custom weights configuration exported.');
        });
        
        // Import
        importInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    ClassifierEngine.importDataset(event.target.result);
                    
                    // Repopulate UI lists based on imported dataset
                    this.rebuildUIFromImportedDataset();
                    alert('Model parameters imported successfully.');
                } catch (error) {
                    console.error('[NeuralSight] Failed to parse weights config JSON:', error);
                    alert('File parse error: Invalid weights dataset format.');
                }
            };
            reader.readAsText(file);
        });

        // Reset
        resetBtn.addEventListener('click', () => {
            if (confirm('Reset Classifier: Are you sure you want to delete all trained classes and dataset samples?')) {
                ClassifierEngine.resetCustomClassifier();
                UIManager.resetUIClassifier();
                console.log('[NeuralSight] Model and UI reset.');
            }
        });
    },

    rebuildUIFromImportedDataset() {
        UIManager.resetUIClassifier();
        
        const map = ClassifierEngine.classNamesMap;
        const dataset = ClassifierEngine.knnClassifierInstance.getClassifierDataset();
        
        // Populate custom classes in UIManager
        Object.keys(map).forEach(classIdStr => {
            const classId = parseInt(classIdStr);
            const className = map[classIdStr];
            
            // Calculate how many samples are in this class tensor
            const tensor = dataset[classIdStr];
            const sampleCount = tensor ? tensor.shape[0] : 0;
            
            const classObj = {
                id: classId,
                name: className,
                count: sampleCount,
                thumbnails: [] // No real-time camera frames available since JSON holds weights only
            };
            
            UIManager.customClasses.push(classObj);
            
            // Remove empty prompt if present
            const prompt = document.getElementById('no-classes-prompt');
            if (prompt) prompt.classList.add('hidden');
            
            // Re-render class card
            UIManager.renderCustomClassCard(classObj);
            
            // Put a preset icon/placeholder in thumbnail rows to indicate loaded state
            const thumbsContainer = document.getElementById(`class-thumbs-${classId}`);
            thumbsContainer.innerHTML = '';
            for (let i = 0; i < Math.min(sampleCount, 3); i++) {
                const icon = document.createElement('i');
                icon.className = 'fa-solid fa-file-shield text-glow-cyan';
                icon.style.fontSize = '24px';
                icon.style.margin = '5px';
                thumbsContainer.appendChild(icon);
            }
            
            // Update counter badge
            document.getElementById(`class-count-${classId}`).textContent = `${sampleCount} samples`;
        });
        
        // Bump classCounter so that any added classes don't conflict with existing IDs
        const existingIds = Object.keys(map).map(id => parseInt(id));
        UIManager.classCounter = existingIds.length > 0 ? Math.max(...existingIds) + 1 : 0;
        
        UIManager.updateCustomToggles();
        console.log('[NeuralSight] Rebuilt visual card list from imported model.');
    },

    /**
     * Start the animation loops that run inferences and canvas preprocessors
     */
    startRenderLoop() {
        const loop = async () => {
            try {
                if (ClassifierEngine.isLoaded) {
                    await this.runLoopCycle();
                }
            } catch (err) {
                console.error('[NeuralSight] Error inside render loop iteration:', err);
            }
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    },

    async runLoopCycle() {
        // ------------------ TAB 1: STANDARD CLASSIFIER INFERENCE ------------------
        if (UIManager.activeTab === 'tab-standard' && !this.isPredictingStandard) {
            let elementToClassify = null;
            
            if (UIManager.activeSource === 'webcam' && UIManager.isWebcamActive) {
                elementToClassify = document.getElementById('webcam');
            } else if (UIManager.activeSource === 'upload' && UIManager.activeImageElement) {
                elementToClassify = UIManager.activeImageElement;
            } else if (UIManager.activeSource === 'gallery' && UIManager.activeImageElement) {
                elementToClassify = UIManager.activeImageElement;
            }
            
            if (elementToClassify) {
                this.isPredictingStandard = true;
                const startTime = performance.now();
                
                try {
                    // 1. Run standard pre-trained MobileNet inference
                    let predictions = await ClassifierEngine.predictStandard(elementToClassify);
                    
                    // 2. Query custom models if trained categories exist
                    if (ClassifierEngine.numClasses > 0) {
                        const customResult = await ClassifierEngine.predictCustom(elementToClassify);
                        if (customResult && customResult.predictions) {
                            // Map custom categories to prediction models format with tag isCustom
                            // Filter out predictions where probability is 0 (image does not resemble class)
                            const blendedCustom = customResult.predictions
                                .filter(pred => pred.probability > 0)
                                .map(pred => ({
                                    className: pred.className,
                                    probability: pred.probability,
                                    isCustom: true
                                }));
                            
                            // Combine standard and custom arrays
                            predictions = [...blendedCustom, ...predictions];
                            
                            // Re-sort blended output by probability scores descending
                            predictions.sort((a, b) => b.probability - a.probability);
                        }
                    }
                    
                    const latency = Math.round(performance.now() - startTime);
                    UIManager.drawPredictions(predictions, latency, elementToClassify);
                } catch (e) {
                    console.error('[NeuralSight] Standard classification failed:', e);
                } finally {
                    this.isPredictingStandard = false;
                }
            }
        }
        
        // ------------------ TAB 2: CUSTOM MODEL INFERENCE ------------------
        if (UIManager.activeTab === 'tab-custom' && !this.isPredictingCustom) {
            const livePredictToggle = document.getElementById('toggle-custom-predict');
            
            if (livePredictToggle && livePredictToggle.checked && UIManager.isWebcamActive) {
                this.isPredictingCustom = true;
                const video = document.getElementById('custom-feed-video');
                const startTime = performance.now();
                
                try {
                    const result = await ClassifierEngine.predictCustom(video);
                    const latency = Math.round(performance.now() - startTime);
                    
                    if (result) {
                        UIManager.drawCustomPredictions(result, latency);
                    }
                } catch (e) {
                    console.error('[NeuralSight] Custom transfer-learning inference failed:', e);
                } finally {
                    this.isPredictingCustom = false;
                }
            }
        }

        // ------------------ TAB 3: NEURAL X-RAY CANVAS RENDERING ------------------
        if (UIManager.activeTab === 'tab-xray') {
            let activeElement = null;
            
            // Read standard tab settings to grab correct active preview target
            if (UIManager.activeSource === 'webcam' && UIManager.isWebcamActive) {
                activeElement = document.getElementById('xray-webcam') || document.getElementById('webcam');
            } else if (UIManager.activeSource === 'upload' && UIManager.activeImageElement) {
                activeElement = UIManager.activeImageElement;
            } else if (UIManager.activeSource === 'gallery' && UIManager.activeImageElement) {
                activeElement = UIManager.activeImageElement;
            }
            
            const activeFilter = document.querySelector('input[name="xray-filter"]:checked').value;
            UIManager.processXRayFilters(activeElement, activeFilter);
        }
    }
};

// Start system on page load
window.addEventListener('DOMContentLoaded', () => {
    App.init();
});
