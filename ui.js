/**
 * ==========================================================================
 * NEURALSIGHT UI PRESENTATION LAYER (ui.js)
 * Manages tabs, inputs, webcam, canvas filtering (Sobel), logs, and charts
 * ==========================================================================
 */

const UIManager = {
    // Current Active Tab and Input Source
    activeTab: 'tab-standard',
    activeSource: 'webcam', // 'webcam', 'upload', 'gallery'
    
    // Webcam stream
    stream: null,
    isWebcamActive: false,
    
    // Performance Ticks Queue
    latencyQueue: [],
    maxLatencyHistory: 15,
    minLatency: Infinity,
    maxLatency: -Infinity,
    totalLatencySum: 0,
    totalPredictionsCount: 0,
    
    // Custom Classes Counter
    classCounter: 0,
    customClasses: [], // Array of objects: { id, name, count, thumbnails: [] }
    recordingClassId: null,
    recordingInterval: null,
    
    // Image element for upload/preset sources
    activeImageElement: null,
    
    // Grad-CAM Visualization States
    currentGradCAM: null,
    isComputingGradCAM: false,
    lastGradCAMTime: 0,
    lastGradCAMTargetClass: -1,
    lastGradCAMTargetLabel: '',
    
    // Image Captioning Engine States
    captionMode: 'cloud', // 'cloud' or 'local'
    isGeneratingCaption: false,
    captionPipeline: null,
    autoDescribeInterval: null,
    hfApiToken: localStorage.getItem('hf_api_token') || '',

    /**
     * Bind all DOM events, tab actions and configurations
     */
    init() {
        this.bindTabs();
        this.bindSources();
        this.bindModelSelection();
        this.bindWebcam();
        this.bindFileUploads();
        this.bindCustomClasses();
        this.bindAnalytics();
        this.bindCaptioning();
        
        // Start memory diagnostic loop
        setInterval(() => this.updateMemoryDiagnostics(), 2000);
    },

    /**
     * Coordinate Tabbed Navigation Controls
     */
    bindTabs() {
        const tabs = document.querySelectorAll('.nav-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetTab = tab.getAttribute('data-tab');
                
                // Toggle tabs CSS
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                // Toggle sections visibility
                document.querySelectorAll('.tab-content').forEach(section => {
                    section.classList.add('hidden');
                    section.classList.remove('active');
                });
                
                const activeSection = document.getElementById(targetTab);
                activeSection.classList.remove('hidden');
                setTimeout(() => activeSection.classList.add('active'), 50);
                
                this.activeTab = targetTab;
                console.log(`[NeuralSight] Navigation changed to tab: ${targetTab}`);
                
                // Synchronize webcam element source when shifting to/from tabs
                this.syncTabWebcams();
            });
        });
    },

    /**
     * Switch input sources: Live Webcam, Upload file, Gallery presets
     */
    bindSources() {
        const pills = document.querySelectorAll('.selector-pill:not([data-model])');
        pills.forEach(pill => {
            pill.addEventListener('click', () => {
                const source = pill.getAttribute('data-source');
                if (!source) return;
                
                // CSS toggling
                pills.forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                
                // Viewports toggling
                document.getElementById('webcam-viewport').classList.add('hidden');
                document.getElementById('upload-viewport').classList.add('hidden');
                document.getElementById('gallery-viewport').classList.add('hidden');
                
                document.getElementById(`${source}-viewport`).classList.remove('hidden');
                this.activeSource = source;
                console.log(`[NeuralSight] Active source set to: ${source}`);
                
                // Reset predictions when switching sources
                this.resetPredictionOutput();
                this.clearAllOverlays();
                
                if (source === 'webcam') {
                    this.startWebcam();
                } else {
                    this.stopWebcam();
                    
                    if (source === 'gallery') {
                        this.clearGalleryPreview();
                    }
                }
            });
        });

        // Bind preset gallery item selects
        const galleryCards = document.querySelectorAll('.gallery-card');
        galleryCards.forEach(card => {
            card.addEventListener('click', () => {
                galleryCards.forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                
                const imgUrl = card.getAttribute('data-url');
                this.loadPresetImage(imgUrl);
                
                // Show preset preview page beautifully
                const previewImg = document.getElementById('gallery-preview');
                previewImg.src = imgUrl;
                
                document.getElementById('gallery-grid-container').classList.add('hidden');
                document.getElementById('gallery-preview-container').classList.remove('hidden');
            });
        });

        // Bind Gallery preview back button
        const clearGalleryBtn = document.getElementById('btn-clear-gallery');
        clearGalleryBtn.addEventListener('click', () => {
            this.clearGalleryPreview();
        });
    },

    /**
     * Start/Stop and Sync Webcams
     */
    bindWebcam() {
        document.getElementById('btn-enable-webcam').addEventListener('click', () => {
            this.startWebcam();
        });
    },

    async startWebcam() {
        if (this.isWebcamActive) return;
        
        const videoElement = document.getElementById('webcam');
        const customVideoElement = document.getElementById('custom-feed-video');
        const xrayVideoElement = document.getElementById('xray-webcam');
        
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 320, height: 240, facingMode: 'user' },
                audio: false
            });
            
            videoElement.srcObject = this.stream;
            customVideoElement.srcObject = this.stream;
            if (xrayVideoElement) xrayVideoElement.srcObject = this.stream;
            
            this.isWebcamActive = true;
            document.getElementById('webcam-overlay-prompt').classList.add('hidden');
            console.log('[NeuralSight] Webcam stream activated successfully.');
        } catch (error) {
            console.error('[NeuralSight] Failed to load webcam feed:', error);
            document.getElementById('webcam-overlay-prompt').classList.remove('hidden');
            this.isWebcamActive = false;
        }
    },

    stopWebcam() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.isWebcamActive = false;
        console.log('[NeuralSight] Webcam feed stopped.');
    },

    syncTabWebcams() {
        // Stop cameras if user navigates to Analytics, reactivate for classifier and X-Ray tabs
        const needsWebcam = (this.activeTab === 'tab-standard' && this.activeSource === 'webcam') || 
                            (this.activeTab === 'tab-custom') || 
                            (this.activeTab === 'tab-xray' && this.activeSource === 'webcam');
                            
        if (needsWebcam) {
            this.startWebcam();
        } else {
            // Wait brief moment before stopping to allow layout to settle
            setTimeout(() => {
                const stillNeedsWebcam = (this.activeTab === 'tab-standard' && this.activeSource === 'webcam') || 
                                         (this.activeTab === 'tab-custom') || 
                                         (this.activeTab === 'tab-xray' && this.activeSource === 'webcam');
                if (!stillNeedsWebcam) {
                    this.stopWebcam();
                }
            }, 100);
        }
    },

    /**
     * Manage File Uploading and Drag & Drop Viewports
     */
    bindFileUploads() {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const clearBtn = document.getElementById('btn-clear-upload');
        
        // Drag events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add('hover');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove('hover');
            }, false);
        });
        
        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) {
                this.handleUploadedFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (fileInput.files.length) {
                this.handleUploadedFile(fileInput.files[0]);
            }
        });
        
        clearBtn.addEventListener('click', () => {
            this.clearUploadedFile();
        });
    },

    handleUploadedFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Format error: Please drag and drop an image file (PNG/JPG)');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.activeImageElement = img;
                
                const previewImg = document.getElementById('upload-preview');
                previewImg.src = e.target.result;
                
                document.getElementById('drop-zone').classList.add('hidden');
                document.getElementById('upload-preview-container').classList.remove('hidden');
                
                console.log(`[NeuralSight] Loaded user image file: ${file.name}`);
                this.resetPredictionOutput();
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    },

    clearUploadedFile() {
        this.activeImageElement = null;
        document.getElementById('file-input').value = '';
        document.getElementById('upload-preview').src = '';
        document.getElementById('upload-preview-container').classList.add('hidden');
        document.getElementById('drop-zone').classList.remove('hidden');
        this.resetPredictionOutput();
        this.clearAllOverlays();
        console.log('[NeuralSight] Cleared user uploaded image.');
    },

    loadPresetImage(url) {
        const img = new Image();
        img.crossOrigin = 'anonymous'; // Enable CORS retrieval for Unsplash CDNs
        img.onload = () => {
            this.activeImageElement = img;
            console.log(`[NeuralSight] Preset gallery image loaded successfully: ${url}`);
            this.resetPredictionOutput();
        };
        img.onerror = () => {
            console.error(`[NeuralSight] Failed to load preset image from URL: ${url}`);
            alert('CDN CORS blockage: Error loading image from Unsplash. Try uploading a local file instead.');
        };
        img.src = url;
    },

    /**
     * Render prediction states
     */
    drawPredictions(predictions, latency, elementToClassify) {
        if (!predictions || predictions.length === 0) {
            this.clearAllOverlays();
            return;
        }
        
        // Draw COCO-SSD bounding box overlays
        if (ClassifierEngine.activeModelId === 'cocossd' && elementToClassify) {
            this.drawBoundingBoxes(predictions, elementToClassify);
        } else {
            this.clearAllOverlays();
        }
        
        // Measure FPS and latency
        this.logLatencyMetrics(latency);
        
        // Main top prediction card
        const topPred = predictions[0];
        const topPct = Math.round(topPred.probability * 100);
        
        const topLabel = document.getElementById('top-pred-label');
        topLabel.innerHTML = topPred.className;
        
        // Visual indicator badge if prediction is a custom user-trained category
        if (topPred.isCustom) {
            topLabel.innerHTML += ` <span class="tag-custom-badge"><i class="fa-solid fa-graduation-cap"></i> Tagged</span>`;
        }
        
        document.getElementById('top-pred-pct').textContent = `${topPct}%`;
        document.getElementById('top-pred-bar').style.width = `${topPct}%`;
        
        // Secondary Predictions rows
        const secondaryContainer = document.getElementById('secondary-predictions');
        secondaryContainer.innerHTML = '';
        
        for (let i = 1; i < predictions.length; i++) {
            const pred = predictions[i];
            const pct = Math.round(pred.probability * 100);
            
            const customBadge = pred.isCustom ? ` <span class="tag-custom-badge"><i class="fa-solid fa-graduation-cap"></i> Tagged</span>` : '';
            
            const row = document.createElement('div');
            row.className = 'prediction-item';
            row.innerHTML = `
                <div class="item-meta">
                    <span class="item-label">${pred.className}${customBadge}</span>
                    <span class="item-pct">${pct}%</span>
                </div>
                <div class="progress-bar-wrapper slim">
                    <div class="progress-bar glow-purple" style="width: ${pct}%"></div>
                </div>
            `;
            secondaryContainer.appendChild(row);
        }
        
        // Log to table records (once every 45 ticks to avoid flooding the list on live camera)
        if (this.totalPredictionsCount % 45 === 0) {
            this.addPredictionToLogTable(this.activeSource, topPred.className, topPct, latency);
        }
        this.totalPredictionsCount++;
    },

    drawCustomPredictions(data, latency) {
        if (!data) return;
        
        this.logLatencyMetrics(latency);
        this.totalPredictionsCount++;
        
        const topPred = data.predictions[0];
        const topPct = Math.round(topPred.probability * 100);
        
        // Floating preview indicator
        document.getElementById('custom-prediction-floating').textContent = `${data.label} (${topPct}%)`;
        
        // Top Card custom predict
        document.getElementById('custom-top-label').textContent = data.label;
        document.getElementById('custom-top-pct').textContent = `${topPct}%`;
        document.getElementById('custom-top-bar').style.width = `${topPct}%`;
        
        // Secondary Custom Predictions
        const secondaryContainer = document.getElementById('custom-secondary-predictions');
        secondaryContainer.innerHTML = '';
        
        for (let i = 1; i < data.predictions.length; i++) {
            const pred = data.predictions[i];
            const pct = Math.round(pred.probability * 100);
            
            const row = document.createElement('div');
            row.className = 'prediction-item';
            row.innerHTML = `
                <div class="item-meta">
                    <span class="item-label">${pred.className}</span>
                    <span class="item-pct">${pct}%</span>
                </div>
                <div class="progress-bar-wrapper slim">
                    <div class="progress-bar glow-cyan" style="width: ${pct}%"></div>
                </div>
            `;
            secondaryContainer.appendChild(row);
        }
        
        // Log to table occasionally
        if (this.totalPredictionsCount % 45 === 0) {
            this.addPredictionToLogTable('webcam (Custom Model)', data.label, topPct, latency);
        }
    },

    resetPredictionOutput() {
        this.currentGradCAM = null;
        document.getElementById('top-pred-label').textContent = 'Awaiting Input...';
        document.getElementById('top-pred-pct').textContent = '0%';
        document.getElementById('top-pred-bar').style.width = '0%';
        document.getElementById('secondary-predictions').innerHTML = `
            <div class="prediction-item">
                <div class="item-meta">
                    <span class="item-label">Prediction #2</span>
                    <span class="item-pct">0%</span>
                </div>
                <div class="progress-bar-wrapper slim">
                    <div class="progress-bar" style="width: 0%"></div>
                </div>
            </div>
        `;
    },

    /**
     * Custom Classes Dashboard Builder (Transfer Learning View)
     */
    bindCustomClasses() {
        const addBtn = document.getElementById('btn-add-class');
        const input = document.getElementById('custom-class-name');
        
        addBtn.addEventListener('click', () => {
            const name = input.value.trim();
            if (name) {
                this.addNewCustomClass(name);
                input.value = '';
            }
        });
        
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                addBtn.click();
            }
        });
        
        // Dynamic hookups for the new Labeled Batch Upload Tagger
        const batchUpload = document.getElementById('batch-media-upload');
        const batchLabel = document.getElementById('batch-upload-label');
        const batchInput = document.getElementById('batch-media-label');
        const batchTrainBtn = document.getElementById('btn-batch-train');
        
        batchUpload.addEventListener('change', () => {
            if (batchUpload.files.length > 0) {
                batchLabel.innerHTML = `<i class="fa-solid fa-file-circle-check"></i> ${batchUpload.files.length} Selected`;
            } else {
                batchLabel.innerHTML = `<i class="fa-solid fa-images"></i> Select Images`;
            }
        });
        
        batchTrainBtn.addEventListener('click', () => {
            const files = batchUpload.files;
            const label = batchInput.value.trim();
            
            if (files.length === 0) {
                alert('Input error: Please select training images/videos first.');
                return;
            }
            
            if (!label) {
                alert('Input error: Please enter a label tag (e.g. "soda").');
                return;
            }
            
            // Check if class already exists, if not, create it
            let classObj = this.customClasses.find(c => c.name.toLowerCase() === label.toLowerCase());
            
            if (!classObj) {
                this.addNewCustomClass(label);
                classObj = this.customClasses.find(c => c.name.toLowerCase() === label.toLowerCase());
            }
            
            // Trigger batch tagged preprocessing training
            this.processUploadedTrainingFiles(classObj.id, Array.from(files));
            
            // Reset input slots
            batchUpload.value = '';
            batchLabel.innerHTML = `<i class="fa-solid fa-images"></i> Select Images`;
            batchInput.value = '';
        });
    },

    addNewCustomClass(name) {
        const classId = this.classCounter++;
        const classObj = {
            id: classId,
            name: name,
            count: 0,
            thumbnails: []
        };
        
        this.customClasses.push(classObj);
        
        // Remove empty classes prompt
        const prompt = document.getElementById('no-classes-prompt');
        if (prompt) prompt.classList.add('hidden');
        
        this.renderCustomClassCard(classObj);
        this.updateCustomToggles();
        console.log(`[NeuralSight] Added new custom class category: ${name}`);
    },

    renderCustomClassCard(classObj) {
        const container = document.getElementById('class-cards-container');
        
        const card = document.createElement('div');
        card.className = 'class-card';
        card.id = `class-card-${classObj.id}`;
        card.innerHTML = `
            <div class="class-card-header">
                <span class="class-name">${classObj.name}</span>
                <span class="badge-count" id="class-count-${classObj.id}">0 samples</span>
            </div>
            
            <div class="class-thumbnails" id="class-thumbs-${classObj.id}">
                <!-- Captured frames go here -->
            </div>
            
            <div class="class-card-actions">
                <button class="btn-capture-record" id="btn-record-${classObj.id}">
                    <i class="fa-solid fa-circle-dot"></i> Hold to Record
                </button>
                <label for="class-upload-${classObj.id}" class="btn-capture-record btn-secondary-sm cursor-pointer">
                    <i class="fa-solid fa-file-arrow-up"></i> Add Media
                </label>
                <input type="file" id="class-upload-${classObj.id}" accept="image/*,video/*" multiple class="hidden">
                <button class="btn-delete-class" id="btn-delete-${classObj.id}" title="Remove Class">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </div>
        `;
        
        container.appendChild(card);
        
        // Setup Hold to Record Events
        const recordBtn = document.getElementById(`btn-record-${classObj.id}`);
        
        // PC Actions
        recordBtn.addEventListener('mousedown', () => this.startRecordingExamples(classObj.id));
        recordBtn.addEventListener('mouseup', () => this.stopRecordingExamples());
        recordBtn.addEventListener('mouseleave', () => this.stopRecordingExamples());
        
        // Mobile/Tablet supports
        recordBtn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startRecordingExamples(classObj.id);
        });
        recordBtn.addEventListener('touchend', () => this.stopRecordingExamples());
        
        // Setup Batch Media Uploader trigger
        const fileInput = document.getElementById(`class-upload-${classObj.id}`);
        fileInput.addEventListener('change', (e) => {
            if (fileInput.files.length) {
                this.processUploadedTrainingFiles(classObj.id, Array.from(fileInput.files));
                fileInput.value = ''; // Reset input selection
            }
        });
        
        // Delete class action
        document.getElementById(`btn-delete-${classObj.id}`).addEventListener('click', () => {
            this.deleteCustomClass(classObj.id);
        });
    },

    startRecordingExamples(classId) {
        if (!this.isWebcamActive) {
            alert('Camera Offline: Enable the webcam first to record samples.');
            return;
        }
        
        this.recordingClassId = classId;
        console.log(`[NeuralSight] Recording dataset samples for Class ID: ${classId}`);
        
        // Set pulsing badge status
        ClassifierEngine.updateStatus('model-status', 'training', 'Training custom head...');
        
        // Add one snapshot immediately
        this.captureSampleFrame(classId);
        
        // Keep adding frames periodically every 150ms while button is held
        this.recordingInterval = setInterval(() => {
            this.captureSampleFrame(classId);
        }, 150);
    },

    stopRecordingExamples() {
        if (this.recordingInterval) {
            clearInterval(this.recordingInterval);
            this.recordingInterval = null;
            this.recordingClassId = null;
            console.log('[NeuralSight] Stopped recording custom class dataset frames.');
            ClassifierEngine.updateStatus('model-status', 'online', 'Model Ready');
            this.updateCustomToggles();
        }
    },

    captureSampleFrame(classId) {
        const video = document.getElementById('custom-feed-video');
        
        // Draw frame onto a hidden mini canvas to save memory and scale it correctly
        const miniCanvas = document.createElement('canvas');
        miniCanvas.width = 112; // Thin thumbnail
        miniCanvas.height = 112;
        const ctx = miniCanvas.getContext('2d');
        
        // Mirror camera crop frame
        ctx.translate(112, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, 112, 112);
        
        const dataUrl = miniCanvas.toDataURL('image/jpeg', 0.7);
        
        // Find class object and push thumbnails
        const classObj = this.customClasses.find(c => c.id === classId);
        if (classObj) {
            classObj.count++;
            
            // Limit thumbnails shown to last 8 items to avoid DOM overload
            if (classObj.thumbnails.length >= 8) {
                classObj.thumbnails.shift();
            }
            classObj.thumbnails.push(dataUrl);
            
            // Record target training data in classifier.js KNN
            ClassifierEngine.addCustomExample(video, classId, classObj.name);
            
            // Update class card interface
            document.getElementById(`class-count-${classId}`).textContent = `${classObj.count} samples`;
            
            const thumbsContainer = document.getElementById(`class-thumbs-${classId}`);
            thumbsContainer.innerHTML = '';
            classObj.thumbnails.forEach(src => {
                const img = document.createElement('img');
                img.src = src;
                thumbsContainer.appendChild(img);
            });
        }
    },

    deleteCustomClass(classId) {
        this.stopRecordingExamples();
        
        // Wipe class data in KNN
        ClassifierEngine.clearClass(classId);
        
        // Wipe class card from DOM
        document.getElementById(`class-card-${classId}`).remove();
        
        // Filter classes array
        this.customClasses = this.customClasses.filter(c => c.id !== classId);
        
        if (this.customClasses.length === 0) {
            const prompt = document.getElementById('no-classes-prompt');
            if (prompt) prompt.classList.remove('hidden');
        }
        
        this.updateCustomToggles();
        console.log(`[NeuralSight] Deleted class ID: ${classId}`);
    },

    updateCustomToggles() {
        const totalSamples = this.customClasses.reduce((sum, c) => sum + c.count, 0);
        const hasCategories = this.customClasses.length > 0;
        
        // Toggle live predict buttons
        const predictToggle = document.getElementById('toggle-custom-predict');
        const resetBtn = document.getElementById('btn-clear-classifier');
        const exportBtn = document.getElementById('btn-export-model');
        
        if (totalSamples > 0 && this.customClasses.length >= 2) {
            predictToggle.removeAttribute('disabled');
        } else {
            predictToggle.setAttribute('disabled', true);
            predictToggle.checked = false;
        }
        
        if (hasCategories) {
            resetBtn.removeAttribute('disabled');
        } else {
            resetBtn.setAttribute('disabled', true);
        }
        
        if (totalSamples > 0) {
            exportBtn.removeAttribute('disabled');
        } else {
            exportBtn.setAttribute('disabled', true);
        }
    },

    resetUIClassifier() {
        this.stopRecordingExamples();
        
        // Clean DOM
        document.getElementById('class-cards-container').innerHTML = `
            <div class="empty-state-card" id="no-classes-prompt">
                <i class="fa-solid fa-graduation-cap"></i>
                <h3>No Custom Classes Created Yet</h3>
                <p>Enter a label name above to build your own custom neural categories</p>
            </div>
        `;
        
        this.customClasses = [];
        this.classCounter = 0;
        
        // Reset score panels
        document.getElementById('custom-prediction-floating').textContent = 'Inactive';
        document.getElementById('custom-top-label').textContent = 'Awaiting Custom Model...';
        document.getElementById('custom-top-pct').textContent = '0%';
        document.getElementById('custom-top-bar').style.width = '0%';
        document.getElementById('custom-secondary-predictions').innerHTML = '';
        
        this.updateCustomToggles();
    },

    /**
     * Canvas image "X-Ray" filters processors (Standard 224x224 input feed visualizer)
     */
    processXRayFilters(sourceElement, filterType) {
        const canvas = document.getElementById('xray-canvas');
        const ctx = canvas.getContext('2d');
        
        // 1. Draw source crop frame correctly onto 224x224 canvas
        ctx.clearRect(0,0,224,224);
        
        if (this.isWebcamActive && sourceElement.tagName === 'VIDEO') {
            // Mirror camera feed for X-Ray
            ctx.save();
            ctx.translate(224, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(sourceElement, 0, 0, 224, 224);
            ctx.restore();
        } else if (sourceElement) {
            ctx.drawImage(sourceElement, 0, 0, 224, 224);
        } else {
            // Draw visual tech fallback pattern
            ctx.fillStyle = '#05070B';
            ctx.fillRect(0,0,224,224);
            ctx.fillStyle = '#1e293b';
            ctx.font = '10px JetBrains Mono';
            ctx.fillText('NO FEED DETECTED', 62, 116);
            return;
        }
        
        // 2. Grab Image Pixel Data array
        const imgData = ctx.getImageData(0, 0, 224, 224);
        const data = imgData.data;
        
        // 3. Render color distribution metrics (RGB Histogram)
        this.drawHistogram(data);
        
        // 4. Apply image preprocessor matrix filters
        if (filterType !== 'gradcam') {
            const explainCard = document.getElementById('xray-explain-card');
            if (explainCard) explainCard.classList.add('hidden');
        }
        
        if (filterType === 'grayscale') {
            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i+1];
                const b = data[i+2];
                // Luma formula for grayscale
                const gray = 0.299*r + 0.587*g + 0.114*b;
                data[i] = gray;
                data[i+1] = gray;
                data[i+2] = gray;
            }
            ctx.putImageData(imgData, 0, 0);
            
        } else if (filterType === 'edge') {
            // High-speed Sobel Edge Filter over 224x224 matrix
            const width = 224;
            const height = 224;
            const output = ctx.createImageData(width, height);
            const outData = output.data;
            
            // Sobel Convolution Kernels
            const Gx = [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ];
            const Gy = [
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ];
            
            for (let y = 1; y < height - 1; y++) {
                for (let x = 1; x < width - 1; x++) {
                    let sumX = 0;
                    let sumY = 0;
                    
                    // Convolve 3x3 surrounding pixels
                    for (let ky = -1; ky <= 1; ky++) {
                        for (let kx = -1; kx <= 1; kx++) {
                            const pixelIndex = ((y + ky) * width + (x + kx)) * 4;
                            // Grayscale luminance approximation of target pixel
                            const val = 0.3 * data[pixelIndex] + 0.59 * data[pixelIndex+1] + 0.11 * data[pixelIndex+2];
                            
                            sumX += val * Gx[ky+1][kx+1];
                            sumY += val * Gy[ky+1][kx+1];
                        }
                    }
                    
                    const magnitude = Math.sqrt(sumX * sumX + sumY * sumY);
                    const idx = (y * width + x) * 4;
                    
                    // Glow color outlines (Ultraviolet edge coloring style)
                    outData[idx] = magnitude * 0.7;     // Red channel
                    outData[idx+1] = magnitude * 0.9;   // Green channel
                    outData[idx+2] = magnitude;         // Blue channel (Neon vibe)
                    outData[idx+3] = 255;               // Solid Alpha
                }
            }
            ctx.putImageData(output, 0, 0);
            
        } else if (filterType === 'grid') {
            // Matrix Pixel Grid lines overlay
            ctx.strokeStyle = 'rgba(187, 92, 53, 0.18)'; // Cyan Grid lines
            ctx.lineWidth = 1;
            const gridSize = 16;
            
            ctx.beginPath();
            for (let x = 0; x < 224; x += gridSize) {
                ctx.moveTo(x, 0);
                ctx.lineTo(x, 224);
            }
            for (let y = 0; y < 224; y += gridSize) {
                ctx.moveTo(0, y);
                ctx.lineTo(224, y);
            }
            ctx.stroke();
        } else if (filterType === 'gradcam') {
            // 1. Show the dynamic explanation card
            const explainCard = document.getElementById('xray-explain-card');
            if (explainCard) explainCard.classList.remove('hidden');
            
            // 2. Check if we need to launch a new async Grad-CAM computation in background
            if (sourceElement && !this.isComputingGradCAM && (performance.now() - this.lastGradCAMTime > 180)) {
                this.isComputingGradCAM = true;
                
                (async () => {
                    try {
                        // Get top predicted class index and standard label/prob
                        const topClassIndex = await ClassifierEngine.getTopClassIndex(sourceElement);
                        
                        // We also get standard predictions to display a clean classification name
                        const predictions = await ClassifierEngine.predictStandard(sourceElement);
                        const topPred = predictions[0];
                        const displayLabel = topPred ? `${topPred.className} (${Math.round(topPred.probability * 100)}%)` : 'Unknown Prediction';
                        
                        // Compute Grad-CAM heatmap array [224 * 224]
                        const heatmap = await ClassifierEngine.computeGradCAM(sourceElement, topClassIndex);
                        
                        // Save states to UIManager
                        this.currentGradCAM = heatmap;
                        this.lastGradCAMTargetClass = topClassIndex;
                        this.lastGradCAMTargetLabel = displayLabel;
                        this.lastGradCAMTime = performance.now();
                        
                        // Update explanation text
                        const targetLabelElement = document.getElementById('xray-explain-target');
                        if (targetLabelElement) {
                            targetLabelElement.textContent = displayLabel;
                        }
                    } catch (err) {
                        console.error('[NeuralSight] Grad-CAM background run failed:', err);
                    } finally {
                        this.isComputingGradCAM = false;
                    }
                })();
            }
            
            // 3. Draw original image frame, then apply the alpha blended overlay
            if (this.currentGradCAM) {
                const heatmap = this.currentGradCAM;
                for (let i = 0; i < data.length; i += 4) {
                    const pixelIdx = i / 4;
                    const val = heatmap[pixelIdx];
                    
                    const glow = this.getGlowColor(val);
                    const alpha = glow.a;
                    
                    // Alpha blending formula: new = (1 - alpha) * original + alpha * glow
                    data[i] = Math.round((1 - alpha) * data[i] + alpha * glow.r);
                    data[i+1] = Math.round((1 - alpha) * data[i+1] + alpha * glow.g);
                    data[i+2] = Math.round((1 - alpha) * data[i+2] + alpha * glow.b);
                }
                ctx.putImageData(imgData, 0, 0);
                
                // 4. Draw high-tech HUD overlay on the canvas
                ctx.fillStyle = 'rgba(7, 10, 19, 0.75)';
                ctx.fillRect(8, 8, 208, 24);
                ctx.strokeStyle = 'hsla(187, 92%, 53%, 0.4)';
                ctx.lineWidth = 1;
                ctx.strokeRect(8, 8, 208, 24);
                
                ctx.fillStyle = '#ffffff';
                ctx.font = '700 8px "JetBrains Mono"';
                
                const targetText = this.lastGradCAMTargetLabel ? `TARGET: ${this.lastGradCAMTargetLabel.toUpperCase()}` : 'COMPUTING HEATMAP...';
                ctx.fillText(targetText, 14, 23);
            } else {
                // Show a loading text and a scanner sweep animation on original frame
                ctx.fillStyle = 'rgba(7, 10, 19, 0.7)';
                ctx.fillRect(0, 0, 224, 224);
                
                ctx.fillStyle = 'var(--color-cyan)';
                ctx.font = '700 10px "JetBrains Mono"';
                ctx.fillText('SCANNING DECISION MAP...', 40, 116);
                
                // Tech scanner line
                const scanY = Math.round((performance.now() / 4) % 224);
                ctx.fillStyle = 'hsla(187, 92%, 53%, 0.4)';
                ctx.fillRect(0, scanY, 224, 2);
            }
        }
    },

    /**
     * Maps a normalized value in [0, 1] to a gorgeous neon heatmap color
     * Palette: Dark Violet ➔ Neon Magenta ➔ Hot Orange ➔ Glowing Yellow/White
     * @param {number} value 
     */
    getGlowColor(value) {
        let r, g, b, a;
        
        if (value < 0.2) {
            // Dark violet/transparent baseline
            r = Math.round(value * 5 * 80);
            g = 0;
            b = Math.round(value * 5 * 120 + 40);
            a = value * 0.45; // very low opacity for background pixels
        } else if (value < 0.5) {
            // Purple to rich Magenta
            const t = (value - 0.2) / 0.3;
            r = Math.round(80 + t * 175);
            g = 0;
            b = Math.round(160 - t * 40);
            a = 0.45 + t * 0.25; // ramp up opacity
        } else if (value < 0.8) {
            // Magenta to Neon Orange
            const t = (value - 0.5) / 0.3;
            r = 255;
            g = Math.round(t * 120);
            b = Math.round(120 - t * 120);
            a = 0.7 + t * 0.15;
        } else {
            // Neon Orange to Glowing Yellow/White
            const t = (value - 0.8) / 0.2;
            r = 255;
            g = Math.round(120 + t * 135);
            b = Math.round(t * 220);
            a = 0.85 + t * 0.15; // full glowing opacity
        }
        
        return { r, g, b, a };
    },

    drawHistogram(pixelData) {
        const canvas = document.getElementById('histogram-canvas');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0,0,280,120);
        
        // 1. Gather RGB counters
        const rHist = new Array(256).fill(0);
        const gHist = new Array(256).fill(0);
        const bHist = new Array(256).fill(0);
        
        for (let i = 0; i < pixelData.length; i += 4) {
            rHist[pixelData[i]]++;
            gHist[pixelData[i+1]]++;
            bHist[pixelData[i+2]]++;
        }
        
        // 2. Get highest frequency peak to normalize chart scale
        const maxPeak = Math.max(...rHist, ...gHist, ...bHist, 1);
        
        // 3. Draw RGB paths
        this.drawHistogramChannel(ctx, rHist, maxPeak, 'rgba(239, 68, 68, 0.45)'); // Red
        this.drawHistogramChannel(ctx, gHist, maxPeak, 'rgba(34, 197, 94, 0.45)');  // Green
        this.drawHistogramChannel(ctx, bHist, maxPeak, 'rgba(187, 92, 53, 0.55)');  // Cyan
    },

    drawHistogramChannel(ctx, values, maxVal, color) {
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1.5;
        
        ctx.beginPath();
        ctx.moveTo(0, 120);
        
        const stepWidth = 280 / 256;
        for (let i = 0; i < 256; i++) {
            const h = (values[i] / maxVal) * 105; // Max 105px height
            const x = i * stepWidth;
            const y = 120 - h;
            ctx.lineTo(x, y);
        }
        ctx.lineTo(280, 120);
        ctx.closePath();
        ctx.fill();
    },

    /**
     * Analytics, Performance Logs and Chart builders (Tab 4)
     */
    bindAnalytics() {
        document.getElementById('btn-clear-logs').addEventListener('click', () => {
            this.clearAnalyticsStats();
        });
        
        document.getElementById('btn-export-csv').addEventListener('click', () => {
            this.exportLogsCSV();
        });
    },

    logLatencyMetrics(latency) {
        if (latency <= 0) return;
        
        // Push latency tick
        this.latencyQueue.push(latency);
        if (this.latencyQueue.length > this.maxLatencyHistory) {
            this.latencyQueue.shift();
        }
        
        // Recalculate Min, Max, and Avg
        if (latency < this.minLatency) this.minLatency = latency;
        if (latency > this.maxLatency) this.maxLatency = latency;
        
        this.totalLatencySum += latency;
        const totalCount = this.totalPredictionsCount + 1;
        const avg = Math.round(this.totalLatencySum / totalCount);
        
        // Update DOM metrics labels
        document.getElementById('stat-min-latency').textContent = `${this.minLatency}ms`;
        document.getElementById('stat-avg-latency').textContent = `${avg}ms`;
        document.getElementById('stat-max-latency').textContent = `${this.maxLatency}ms`;
        document.getElementById('stat-total-pred').textContent = totalCount;
        
        document.getElementById('fps-display').textContent = `FPS: ${Math.round(1000/latency)} // Latency: ${latency}ms`;
        document.getElementById('inference-time').textContent = `${latency}ms`;
        
        // Update Chart
        this.renderLatencyChartBars();
    },

    renderLatencyChartBars() {
        const container = document.getElementById('latency-bars-chart');
        container.innerHTML = '';
        
        if (this.latencyQueue.length === 0) return;
        
        const localMax = Math.max(...this.latencyQueue, 1);
        
        this.latencyQueue.forEach((ms) => {
            const pctHeight = (ms / localMax) * 100;
            const bar = document.createElement('div');
            bar.className = 'latency-bar-item';
            bar.style.height = `${Math.max(pctHeight, 5)}%`;
            bar.setAttribute('data-value', ms);
            container.appendChild(bar);
        });
    },

    addPredictionToLogTable(source, label, confidence, latency) {
        const tableBody = document.getElementById('logs-table-body');
        
        // Remove empty state if present
        const emptyRow = tableBody.querySelector('.table-empty-row');
        if (emptyRow) emptyRow.remove();
        
        const timestamp = new Date().toLocaleTimeString();
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${timestamp}</td>
            <td><span class="text-glow-cyan">${source}</span></td>
            <td><span class="text-glow-purple">${label}</span></td>
            <td>${confidence}%</td>
            <td>${latency}ms</td>
        `;
        
        // Insert at the beginning of the table
        tableBody.insertBefore(row, tableBody.firstChild);
        
        // Keep table logs capped at 15 items to prevent page bloating
        if (tableBody.children.length > 15) {
            tableBody.lastChild.remove();
        }
    },

    clearAnalyticsStats() {
        this.latencyQueue = [];
        this.minLatency = Infinity;
        this.maxLatency = -Infinity;
        this.totalLatencySum = 0;
        this.totalPredictionsCount = 0;
        
        document.getElementById('stat-min-latency').textContent = '0ms';
        document.getElementById('stat-avg-latency').textContent = '0ms';
        document.getElementById('stat-max-latency').textContent = '0ms';
        document.getElementById('stat-total-pred').textContent = '0';
        document.getElementById('latency-bars-chart').innerHTML = '';
        
        document.getElementById('logs-table-body').innerHTML = `
            <tr class="table-empty-row">
                <td colspan="5">No classifications logged yet. Complete standard or custom runs.</td>
            </tr>
        `;
        console.log('[NeuralSight] Latency statistics and log table cleared.');
    },

    exportLogsCSV() {
        const rows = [];
        const headers = ['Timestamp', 'Target Source', 'Top Prediction', 'Confidence (%)', 'Latency (ms)'];
        rows.push(headers.join(','));
        
        const logRows = document.querySelectorAll('#logs-table-body tr:not(.table-empty-row)');
        
        if (logRows.length === 0) {
            alert('No records logged yet.');
            return;
        }
        
        logRows.forEach(tr => {
            const cols = tr.querySelectorAll('td');
            const data = Array.from(cols).map(td => `"${td.textContent.trim()}"`);
            rows.push(data.join(','));
        });
        
        const csvContent = "data:text/csv;charset=utf-8," + rows.join("\n");
        const encodedUri = encodeURI(csvContent);
        
        const downloadLink = document.createElement("a");
        downloadLink.setAttribute("href", encodedUri);
        downloadLink.setAttribute("download", `NeuralSight_Classification_Logs_${Date.now()}.csv`);
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        console.log('[NeuralSight] Prediction logs exported to CSV.');
    },

    updateMemoryDiagnostics() {
        if (!ClassifierEngine.isLoaded) return;
        
        const metrics = ClassifierEngine.getMemoryMetrics();
        document.getElementById('inference-memory').textContent = `${metrics.numMegabytes} MB`;
    },

    /**
     * ==========================================================================
     * BATCH TRAINING & VIDEO SLICING ENGINE
     * Processes local tagged files and extracts activation frames inside the browser
     * ==========================================================================
     */
    async processUploadedTrainingFiles(classId, files) {
        const classObj = this.customClasses.find(c => c.id === classId);
        if (!classObj) return;
        
        console.log(`[NeuralSight] Tagging batch training uploads for class '${classObj.name}' (${files.length} files)`);
        this.showTrainingLoader('Media Preprocessing', 'Initiating neural dataset parser...');
        
        let processedCount = 0;
        const totalFiles = files.length;
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const progressPct = Math.round((i / totalFiles) * 100);
            
            // Check file categories: Image vs Video
            if (file.type.startsWith('image/')) {
                this.updateTrainingLoader(progressPct, `Tagging image file: '${file.name}' (${i+1}/${totalFiles})`);
                
                try {
                    await new Promise((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onload = (e) => {
                            const img = new Image();
                            img.onload = () => {
                                // Feed activation to KNN model
                                ClassifierEngine.addCustomExample(img, classId, classObj.name);
                                
                                // Register visual thumbnails
                                this.appendThumbnailToClassUI(classObj, e.target.result);
                                processedCount++;
                                resolve();
                            };
                            img.onerror = reject;
                            img.src = e.target.result;
                        };
                        reader.onerror = reject;
                        reader.readAsDataURL(file);
                    });
                } catch (err) {
                    console.error('[NeuralSight] Failed processing custom image tag:', err);
                }
                
            } else if (file.type.startsWith('video/')) {
                this.updateTrainingLoader(progressPct, `Preparing video clip: '${file.name}' (${i+1}/${totalFiles})`);
                
                try {
                    await new Promise((resolve, reject) => {
                        this.extractFramesFromVideoClip(
                            classId, 
                            file, 
                            (frameCanvas, current, total) => {
                                // Called every seek seeked frame
                                const pct = Math.round((current / total) * 100);
                                this.updateTrainingLoader(progressPct, `Slicing video clip: frame ${current}/${total} extracted (${pct}%)`);
                                
                                // Feed frame image data to model
                                ClassifierEngine.addCustomExample(frameCanvas, classId, classObj.name);
                                
                                // Render mini thumbnail
                                const miniSrc = frameCanvas.toDataURL('image/jpeg', 0.6);
                                this.appendThumbnailToClassUI(classObj, miniSrc);
                            },
                            () => {
                                // Slicing complete
                                console.log(`[NeuralSight] Video slicing complete for: ${file.name}`);
                                resolve();
                            }
                        );
                    });
                    processedCount++;
                } catch (err) {
                    console.error('[NeuralSight] Failed to slice training video frames:', err);
                    reject(err);
                }
            }
        }
        
        this.updateTrainingLoader(100, 'Re-aligning classifier vectors...');
        setTimeout(() => {
            this.hideTrainingLoader();
            this.updateCustomToggles();
            console.log(`[NeuralSight] Tagged dataset training completed. Added ${classObj.count} samples to '${classObj.name}'`);
        }, 800);
    },

    appendThumbnailToClassUI(classObj, src) {
        classObj.count++;
        
        if (classObj.thumbnails.length >= 8) {
            classObj.thumbnails.shift();
        }
        classObj.thumbnails.push(src);
        
        // Update DOM Elements
        document.getElementById(`class-count-${classObj.id}`).textContent = `${classObj.count} samples`;
        
        const thumbsContainer = document.getElementById(`class-thumbs-${classObj.id}`);
        thumbsContainer.innerHTML = '';
        classObj.thumbnails.forEach(tSrc => {
            const img = document.createElement('img');
            img.src = tSrc;
            thumbsContainer.appendChild(img);
        });
    },

    /**
     * Programmatic frame extraction seek routine in browser using HTML5 blob seekers
     */
    extractFramesFromVideoClip(classId, file, onFrameExtract, onComplete) {
        const video = document.createElement('video');
        video.muted = true;
        video.playsInline = true;
        
        // Squeeze sizes down for ML efficiency
        video.width = 224;
        video.height = 224;
        
        const videoUrl = URL.createObjectURL(file);
        video.src = videoUrl;
        
        // Hidden canvas wrapper
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        
        video.addEventListener('loadedmetadata', () => {
            const duration = video.duration;
            
            // Slice 4 frames per second of video, capped between 10 and 25 frames
            const fps = 4;
            const totalFrames = Math.max(Math.min(Math.floor(duration * fps), 25), 8);
            const stepSize = duration / totalFrames;
            
            let frameIndex = 0;
            
            const onSeeked = () => {
                ctx.clearRect(0,0,224,224);
                
                // Draw normal unmirrored frame to KNN class examples
                ctx.drawImage(video, 0, 0, 224, 224);
                
                frameIndex++;
                onFrameExtract(canvas, frameIndex, totalFrames);
                
                if (frameIndex < totalFrames) {
                    video.currentTime = frameIndex * stepSize;
                } else {
                    // Tear down elements
                    video.removeEventListener('seeked', onSeeked);
                    URL.revokeObjectURL(videoUrl);
                    video.src = '';
                    onComplete();
                }
            };
            
            video.addEventListener('seeked', onSeeked);
            
            // Prime first seek
            video.currentTime = 0;
        });
        
        video.load();
    },

    /**
     * Modal progress hud handlers
     */
    showTrainingLoader(title, description) {
        const modal = document.getElementById('training-hud-modal');
        document.getElementById('training-hud-title').textContent = title;
        document.getElementById('training-hud-description').textContent = description;
        document.getElementById('training-hud-bar').style.width = '0%';
        document.getElementById('training-hud-counter').textContent = '0%';
        modal.classList.remove('hidden');
    },

    updateTrainingLoader(pct, description) {
        document.getElementById('training-hud-description').textContent = description;
        document.getElementById('training-hud-bar').style.width = `${pct}%`;
        document.getElementById('training-hud-counter').textContent = `${pct}%`;
    },

    hideTrainingLoader() {
        const modal = document.getElementById('training-hud-modal');
        modal.classList.add('hidden');
    },

    /**
     * Multimodal AI Image Captioning System Bindings
     */
    bindCaptioning() {
        const cloudBtn = document.getElementById('engine-cloud-btn');
        const localBtn = document.getElementById('engine-local-btn');
        const generateBtn = document.getElementById('btn-generate-caption');
        const toggleTokenBtn = document.getElementById('btn-toggle-token-input');
        const tokenBox = document.getElementById('hf-token-box');
        const tokenInput = document.getElementById('hf-api-token');
        const saveTokenBtn = document.getElementById('btn-save-hf-token');
        const autoDescribeToggle = document.getElementById('toggle-auto-describe');
        const engineLabel = document.getElementById('caption-engine-label');

        // Populate HF token input from saved state
        if (this.hfApiToken) {
            tokenInput.value = this.hfApiToken;
        }

        // Engine Select Cloud
        cloudBtn.addEventListener('click', () => {
            cloudBtn.classList.add('active');
            localBtn.classList.remove('active');
            this.captionMode = 'cloud';
            engineLabel.textContent = 'ENGINE: CLOUD (SALESFORCE/BLIP)';
            console.log('[NeuralSight] Multimodal engine set to: Cloud API (BLIP)');
        });

        // Engine Select Local
        localBtn.addEventListener('click', () => {
            localBtn.classList.add('active');
            cloudBtn.classList.remove('active');
            this.captionMode = 'local';
            engineLabel.textContent = 'ENGINE: LOCAL (VIT-GPT2 / QUANTIZED)';
            console.log('[NeuralSight] Multimodal engine set to: Local AI (ViT-GPT2)');
        });

        // Toggle API Token visibility
        toggleTokenBtn.addEventListener('click', () => {
            tokenBox.classList.toggle('hidden');
        });

        // Save Token
        saveTokenBtn.addEventListener('click', () => {
            const token = tokenInput.value.trim();
            this.hfApiToken = token;
            localStorage.setItem('hf_api_token', token);
            tokenBox.classList.add('hidden');
            console.log('[NeuralSight] Saved secure Hugging Face API Token.');
        });

        // Generate manual description
        generateBtn.addEventListener('click', () => {
            this.triggerCaptionGeneration();
        });

        // Auto describe toggle switch
        autoDescribeToggle.addEventListener('change', () => {
            if (autoDescribeToggle.checked) {
                console.log('[NeuralSight] Enabled Autoloop describing (every 5 seconds)');
                this.triggerCaptionGeneration(); // Run immediately first
                this.autoDescribeInterval = setInterval(() => {
                    this.triggerCaptionGeneration();
                }, 5000);
            } else {
                console.log('[NeuralSight] Disabled Autoloop describing.');
                if (this.autoDescribeInterval) {
                    clearInterval(this.autoDescribeInterval);
                    this.autoDescribeInterval = null;
                }
            }
        });
    },

    async triggerCaptionGeneration() {
        if (this.isGeneratingCaption) return;
        
        let elementToDescribe = null;
        if (this.activeSource === 'webcam' && this.isWebcamActive) {
            elementToDescribe = document.getElementById('webcam');
        } else if (this.activeSource === 'upload' && this.activeImageElement) {
            elementToDescribe = this.activeImageElement;
        } else if (this.activeSource === 'gallery' && this.activeImageElement) {
            elementToDescribe = this.activeImageElement;
        }

        if (!elementToDescribe) {
            this.updateCaptionOutput('<span style="font-style: italic; color: var(--color-text-muted);">Please load camera feed or image presets first</span>');
            return;
        }

        this.isGeneratingCaption = true;
        this.showCaptionLoader();

        try {
            let caption = '';
            if (this.captionMode === 'cloud') {
                caption = await this.generateCaptionCloud(elementToDescribe);
            } else {
                caption = await this.generateCaptionLocal(elementToDescribe);
            }
            
            // Clean/format generated output text gracefully
            if (caption) {
                const formatted = caption.charAt(0).toUpperCase() + caption.slice(1);
                this.updateCaptionOutput(`"${formatted}."`);
            } else {
                this.updateCaptionOutput('<span style="font-style: italic; color: var(--color-text-muted);">Empty response. Please try again.</span>');
            }
        } catch (err) {
            console.error('[NeuralSight] Multimodal AI captioning failed:', err);
            let errMsg = err.message || 'Generation failed.';
            
            // Check if this is a standard fetch TypeError caused by browser CORS blocking unauthenticated Hugging Face requests
            if (this.captionMode === 'cloud' && !this.hfApiToken && (err.name === 'TypeError' || errMsg.toLowerCase().includes('fetch') || errMsg.toLowerCase().includes('networkerror'))) {
                errMsg = 'Unauthenticated browser requests are restricted by Hugging Face (CORS). Please click the Key icon below and save a valid Hugging Face API Token, or switch to "Local AI (ViT-GPT2)" to run completely offline in your browser.';
            }
            
            this.updateCaptionOutput(`<span style="color: var(--color-red); font-size: 12px; line-height: 1.45; display: block; max-width: 380px; margin: 0 auto; font-weight: 500;">Error: ${errMsg}</span>`);
        } finally {
            this.hideCaptionLoader();
            this.isGeneratingCaption = false;
        }
    },

    async generateCaptionCloud(element) {
        this.updateCaptionLoaderText('Processing image...');
        const imageBlob = await this.getFrameBlob(element);
        
        this.updateCaptionLoaderText('Querying Hugging Face API...');
        const url = 'https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base';
        const headers = {};
        if (this.hfApiToken) {
            headers["Authorization"] = `Bearer ${this.hfApiToken}`;
        }
        
        const response = await fetch(url, {
            method: "POST",
            headers: headers,
            body: imageBlob
        });
        
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            if (response.status === 503) {
                throw new Error('Hugging Face model is loading/warming up. Please try again in 10 seconds.');
            }
            throw new Error(errData.error || `API returned status ${response.status}`);
        }
        
        const result = await response.json();
        return result[0]?.generated_text;
    },

    async generateCaptionLocal(element) {
        if (!window.transformers) {
            throw new Error('Transformers.js CDN failed to load.');
        }

        if (!this.captionPipeline) {
            this.updateCaptionLoaderText('Downloading model (~120MB)...');
            const { pipeline } = window.transformers;
            this.captionPipeline = await pipeline('image-to-text', 'Xenova/vit-gpt2-image-captioning', {
                quantized: true
            });
        }

        this.updateCaptionLoaderText('Running ONNX Local Pipeline...');
        
        // Resize element to 224x224 data URL for local model feeding
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        
        // Mirror if webcam
        if (this.activeSource === 'webcam' && this.isWebcamActive) {
            ctx.translate(224, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(element, 0, 0, 224, 224);
        
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        const result = await this.captionPipeline(dataUrl);
        return result[0]?.generated_text;
    },

    getFrameBlob(element) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            canvas.width = 384; // optimal size for BLIP base
            canvas.height = 384;
            const ctx = canvas.getContext('2d');
            
            // Mirror camera frame if webcam is active
            if (this.activeSource === 'webcam' && this.isWebcamActive) {
                ctx.translate(384, 0);
                ctx.scale(-1, 1);
            }
            
            ctx.drawImage(element, 0, 0, 384, 384);
            canvas.toBlob((blob) => {
                resolve(blob);
            }, 'image/jpeg', 0.95);
        });
    },

    showCaptionLoader() {
        document.getElementById('caption-loader').classList.remove('hidden');
        document.getElementById('caption-output').style.opacity = '0.1';
    },

    hideCaptionLoader() {
        document.getElementById('caption-loader').classList.add('hidden');
        document.getElementById('caption-output').style.opacity = '1';
    },

    updateCaptionLoaderText(text) {
        const textBlock = document.getElementById('caption-output');
        textBlock.innerHTML = `<span style="font-style: italic; color: var(--color-text-muted); font-size: 11px;">${text}</span>`;
    },

    updateCaptionOutput(htmlContent) {
        document.getElementById('caption-output').innerHTML = htmlContent;
    },

    /**
     * Bind AI model selector events
     */
    bindModelSelection() {
        const pills = document.querySelectorAll('#model-selector-pills .selector-pill');
        pills.forEach(pill => {
            pill.addEventListener('click', async () => {
                if (pill.classList.contains('active')) return;
                
                const modelId = pill.getAttribute('data-model');
                
                // Switch pills CSS visually
                pills.forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                
                try {
                    // Instantly wipe existing canvas overlays
                    this.clearAllOverlays();
                    
                    // Load chosen model dynamically
                    await ClassifierEngine.loadModel(modelId);
                    
                    // Reset prediction outputs to await new model predictions
                    this.resetPredictionOutput();
                    console.log(`[NeuralSight] Successfully loaded and switched active AI Model to: ${modelId}`);
                } catch (err) {
                    console.error('[NeuralSight] Failed to dynamically switch AI architectures:', err);
                    alert(`Inference Engine Switch Failed: Could not load the ${modelId} model.`);
                }
            });
        });
    },

    /**
     * Resets gallery view, hiding the preview container and resetting metrics
     */
    clearGalleryPreview() {
        this.activeImageElement = null;
        const previewContainer = document.getElementById('gallery-preview-container');
        const gridContainer = document.getElementById('gallery-grid-container');
        
        if (previewContainer) previewContainer.classList.add('hidden');
        if (gridContainer) gridContainer.classList.remove('hidden');
        
        document.getElementById('gallery-preview').src = '';
        
        const galleryCards = document.querySelectorAll('.gallery-card');
        galleryCards.forEach(c => c.classList.remove('selected'));
        
        this.resetPredictionOutput();
        this.clearAllOverlays();
        console.log('[NeuralSight] Exited preset gallery preview mode.');
    },

    /**
     * Clears all overlay canvases in the viewports
     */
    clearAllOverlays() {
        const overlays = ['webcam-overlay', 'upload-overlay', 'gallery-overlay'];
        overlays.forEach(id => {
            const canvas = document.getElementById(id);
            if (canvas) {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        });
    },

    /**
     * Renders neon cyan bounding box HUD overlays for COCO-SSD object detections
     */
    drawBoundingBoxes(predictions, element) {
        let canvasId = '';
        if (this.activeSource === 'webcam') canvasId = 'webcam-overlay';
        else if (this.activeSource === 'upload') canvasId = 'upload-overlay';
        else if (this.activeSource === 'gallery') canvasId = 'gallery-overlay';
        
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Sync coordinate space dimensions of overlay canvas with the source element
        const elementWidth = element.videoWidth || element.naturalWidth || element.width || canvas.clientWidth;
        const elementHeight = element.videoHeight || element.naturalHeight || element.height || canvas.clientHeight;
        
        if (canvas.width !== elementWidth || canvas.height !== elementHeight) {
            canvas.width = elementWidth;
            canvas.height = elementHeight;
        }
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!predictions || predictions.length === 0) return;
        
        const isWebcam = (this.activeSource === 'webcam');
        
        predictions.forEach(pred => {
            if (!pred.bbox) return;
            const [x, y, width, height] = pred.bbox;
            const score = Math.round(pred.probability * 100);
            const className = pred.className;
            
            // horizontal webcam mirroring math to avoid backward text
            const drawX = isWebcam ? (elementWidth - x - width) : x;
            
            // Draw Neon Bounding Box
            ctx.strokeStyle = 'hsla(187, 92%, 53%, 0.85)';
            ctx.lineWidth = Math.max(2, Math.round(elementWidth / 200));
            ctx.strokeRect(drawX, y, width, height);
            
            // Bounding Box glow drop-shadow
            ctx.strokeStyle = 'rgba(187, 92, 53, 0.15)';
            ctx.lineWidth = Math.max(6, Math.round(elementWidth / 80));
            ctx.strokeRect(drawX, y, width, height);
            
            // Label box scaling
            const fontSize = Math.max(10, Math.round(elementWidth / 40));
            ctx.font = `700 ${fontSize}px "JetBrains Mono"`;
            const labelText = `${className} (${score}%)`;
            const textWidth = ctx.measureText(labelText).width;
            const padding = Math.max(4, Math.round(fontSize / 3));
            const bannerHeight = fontSize + padding * 2;
            
            // Label box background
            ctx.fillStyle = 'rgba(7, 10, 19, 0.85)';
            ctx.fillRect(drawX, y - bannerHeight, textWidth + padding * 2, bannerHeight);
            
            ctx.strokeStyle = 'hsla(187, 92%, 53%, 0.5)';
            ctx.lineWidth = 1;
            ctx.strokeRect(drawX, y - bannerHeight, textWidth + padding * 2, bannerHeight);
            
            // Draw text tag
            ctx.fillStyle = '#ffffff';
            ctx.fillText(labelText, drawX + padding, y - padding);
        });
    }
};

// Global Exposure
window.UIManager = UIManager;
