/**
 * Facial Capture Module
 * Real-time facial expression analysis during health assessment
 */

console.log('🎭 facial_capture.js loaded');

class FacialCapture {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.stream = null;
        this.captureInterval = null;
        this.renderInterval = null;
        this.isCapturing = false;
        this.frameCount = 0;
        this.facialScores = [];
        this.lastFaceMesh = null; // Store last received face mesh data
        
        // UI elements
        this.container = null;
        this.statusText = null;
        this.scoreDisplay = null;
        
        // CRITICAL: Load existing frames from previous pages
        this.loadPreviousFrames();
    }
    
    /**
     * Load frames captured on previous pages (assessment -> voice transition)
     */
    loadPreviousFrames() {
        const previousData = sessionStorage.getItem('facial_analysis_accumulator');
        if (previousData) {
            try {
                const parsed = JSON.parse(previousData);
                if (parsed.all_frames && Array.isArray(parsed.all_frames)) {
                    this.facialScores = parsed.all_frames;
                    this.frameCount = this.facialScores.length;
                    console.log(`✅ Loaded ${this.frameCount} previous frames from session storage`);
                }
            } catch (e) {
                console.error('❌ Error loading previous frames:', e);
            }
        }
    }
    
    /**
     * Initialize the facial capture widget
     */
    async init() {
        try {
            // Create UI container
            this.createUI();
            
            // Log continuation status
            if (this.frameCount > 0) {
                console.log(`🔄 Facial capture CONTINUING with ${this.frameCount} existing frames`);
                console.log(`📊 Previous averages loaded - will accumulate new frames`);
            } else {
                console.log(`🆕 Facial capture STARTING fresh - no previous frames`);
            }
            
            // Request webcam access
            await this.startWebcam();
            
            // Start capturing frames
            this.startCapture();
            
            this.updateStatus('✅ Facial analysis active', 'success');
        } catch (error) {
            console.error('Facial capture initialization failed:', error);
            this.updateStatus('❌ Webcam unavailable', 'error');
        }
    }
    
    /**
     * Create the UI widget
     * Check if widget already exists from previous page (for seamless continuation)
     */
    createUI() {
        // Check if widget already exists (persisted from previous page)
        const existingWidget = document.getElementById('facial-capture-widget');
        
        if (existingWidget) {
            console.log('♻️ Reusing existing facial capture widget (no refresh!)');
            this.container = existingWidget;
            
            // Get existing video and canvas references
            this.video = document.getElementById('facial-video');
            this.canvas = document.getElementById('facial-canvas');
            this.ctx = this.canvas.getContext('2d');
            this.statusText = document.getElementById('facial-status');
            
            // Update status to show continuation
            this.updateStatus('✅ Continuing analysis', 'success');
            
            return; // Don't create new widget
        }
        
        // Create new container (first time)
        this.container = document.createElement('div');
        this.container.id = 'facial-capture-widget';
        this.container.innerHTML = `
            <div class="facial-widget-header">
                <h3>🎭 Facial Analysis</h3>
                <span class="status-badge" id="facial-status">Initializing...</span>
            </div>
            <div class="video-container">
                <video id="facial-video" autoplay playsinline muted></video>
                <canvas id="facial-canvas"></canvas>
            </div>
            <div class="score-display" id="score-display">
                <div class="score-item">
                    <span class="score-label">Pain</span>
                    <span class="score-value" id="pain-score">-</span>
                </div>
                <div class="score-item">
                    <span class="score-label">Stress</span>
                    <span class="score-value" id="stress-score">-</span>
                </div>
                <div class="score-item">
                    <span class="score-label">Anxiety</span>
                    <span class="score-value" id="anxiety-score">-</span>
                </div>
            </div>
            <div class="frame-counter">
                Frames analyzed: <span id="frame-count">0</span>
                <span id="continuation-indicator" style="display: none; color: #10b981; font-size: 0.85em; margin-left: 8px;">
                    📊 (Continuing from previous page)
                </span>
            </div>
        `;
        
        // Add to page (top-right, fixed position)
        document.body.appendChild(this.container);
        
        // Get references
        this.video = document.getElementById('facial-video');
        this.canvas = document.getElementById('facial-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.statusText = document.getElementById('facial-status');
        
        // Show continuation indicator if we loaded previous frames
        if (this.frameCount > 0) {
            const indicator = document.getElementById('continuation-indicator');
            if (indicator) {
                indicator.style.display = 'inline';
            }
            // Update frame counter to show accumulated count
            const frameCountEl = document.getElementById('frame-count');
            if (frameCountEl) {
                frameCountEl.textContent = this.frameCount;
            }
        }
        
        // Add CSS
        this.injectStyles();
    }
    
    /**
     * Inject widget styles
     */
    injectStyles() {
        const styleId = 'facial-capture-styles';
        if (document.getElementById(styleId)) return;
        
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            #facial-capture-widget {
                position: fixed;
                top: 20px;
                right: 20px;
                width: 320px;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                padding: 15px;
                z-index: 1000;
                animation: slideInRight 0.5s ease-out;
            }
            
            @keyframes slideInRight {
                from {
                    opacity: 0;
                    transform: translateX(100px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            .facial-widget-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
                padding-bottom: 10px;
                border-bottom: 2px solid #e2e8f0;
            }
            
            .facial-widget-header h3 {
                margin: 0;
                font-size: 1.1em;
                color: #075985;
            }
            
            .status-badge {
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 0.85em;
                font-weight: 600;
                background: #e2e8f0;
                color: #64748b;
            }
            
            .status-badge.success {
                background: #d1fae5;
                color: #065f46;
            }
            
            .status-badge.error {
                background: #fee2e2;
                color: #991b1b;
            }
            
            .video-container {
                position: relative;
                width: 100%;
                height: 240px;
                background: #1e293b;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 10px;
            }
            
            #facial-video {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            
            #facial-canvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
            }
            
            .score-display {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                margin-bottom: 10px;
            }
            
            .score-item {
                background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
                padding: 10px;
                border-radius: 8px;
                text-align: center;
            }
            
            .score-label {
                display: block;
                font-size: 0.8em;
                color: #075985;
                font-weight: 600;
                margin-bottom: 5px;
            }
            
            .score-value {
                display: block;
                font-size: 1.4em;
                font-weight: bold;
                color: #0ea5e9;
            }
            
            .frame-counter {
                text-align: center;
                font-size: 0.85em;
                color: #64748b;
            }
            
            @media (max-width: 768px) {
                #facial-capture-widget {
                    position: static;
                    width: 100%;
                    margin-bottom: 20px;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    /**
     * Start webcam stream
     */
    async startWebcam() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = this.stream;
            
            // Wait for video to be ready
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    // Set canvas size to match video
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    resolve();
                };
            });
        } catch (error) {
            throw new Error('Webcam access denied or unavailable');
        }
    }
    
    /**
     * Start capturing and analyzing frames
     */
    startCapture() {
        this.isCapturing = true;
        
        // Render video at 30fps for smooth display
        this.renderInterval = setInterval(() => {
            this.renderFrame();
        }, 1000 / 30); // 30 FPS rendering
        
        // Capture and analyze frames every 333ms (3 FPS to reduce server load)
        this.captureInterval = setInterval(() => {
            this.captureAndAnalyze();
        }, 333); // 3 FPS for backend analysis
    }
    
    /**
     * Render video frame with face mesh overlay (30fps)
     */
    renderFrame() {
        if (!this.isCapturing || !this.video || this.video.paused) return;
        
        // Draw current video frame
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Draw face mesh overlay if available
        if (this.lastFaceMesh && this.lastFaceMesh.landmarks) {
            this.drawFaceMesh(this.lastFaceMesh.landmarks);
        }
    }
    
    /**
     * Draw MediaPipe face mesh on canvas
     */
    drawFaceMesh(landmarks) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Draw facial landmarks
        ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.4)';
        ctx.lineWidth = 1;
        
        // Draw points
        landmarks.forEach(point => {
            const x = point.x * width;
            const y = point.y * height;
            
            ctx.beginPath();
            ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        // Draw facial contours (selected connections for clarity)
        const connections = [
            // Face oval
            ...this.getFaceOvalConnections(),
            // Left eye
            ...this.getEyeConnections(true),
            // Right eye
            ...this.getEyeConnections(false),
            // Lips
            ...this.getLipsConnections(),
            // Eyebrows
            ...this.getEyebrowConnections()
        ];
        
        connections.forEach(([start, end]) => {
            if (landmarks[start] && landmarks[end]) {
                ctx.beginPath();
                ctx.moveTo(landmarks[start].x * width, landmarks[start].y * height);
                ctx.lineTo(landmarks[end].x * width, landmarks[end].y * height);
                ctx.stroke();
            }
        });
    }
    
    /**
     * Get face oval connections (simplified)
     */
    getFaceOvalConnections() {
        // MediaPipe face mesh oval indices (partial for clarity)
        const oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10];
        const connections = [];
        for (let i = 0; i < oval.length - 1; i++) {
            connections.push([oval[i], oval[i + 1]]);
        }
        return connections;
    }
    
    /**
     * Get eye connections
     */
    getEyeConnections(isLeft) {
        if (isLeft) {
            // Left eye outline
            const leftEye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33];
            const connections = [];
            for (let i = 0; i < leftEye.length - 1; i++) {
                connections.push([leftEye[i], leftEye[i + 1]]);
            }
            return connections;
        } else {
            // Right eye outline
            const rightEye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362];
            const connections = [];
            for (let i = 0; i < rightEye.length - 1; i++) {
                connections.push([rightEye[i], rightEye[i + 1]]);
            }
            return connections;
        }
    }
    
    /**
     * Get lips connections
     */
    getLipsConnections() {
        // Outer lips
        const outerLips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61];
        const innerLips = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78];
        
        const connections = [];
        for (let i = 0; i < outerLips.length - 1; i++) {
            connections.push([outerLips[i], outerLips[i + 1]]);
        }
        for (let i = 0; i < innerLips.length - 1; i++) {
            connections.push([innerLips[i], innerLips[i + 1]]);
        }
        return connections;
    }
    
    /**
     * Get eyebrow connections
     */
    getEyebrowConnections() {
        // Left eyebrow
        const leftBrow = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107];
        // Right eyebrow
        const rightBrow = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336];
        
        const connections = [];
        for (let i = 0; i < leftBrow.length - 1; i++) {
            connections.push([leftBrow[i], leftBrow[i + 1]]);
        }
        for (let i = 0; i < rightBrow.length - 1; i++) {
            connections.push([rightBrow[i], rightBrow[i + 1]]);
        }
        return connections;
    }
    
    /**
     * Capture frame and send to backend for analysis
     */
    async captureAndAnalyze() {
        if (!this.isCapturing) return;
        
        try {
            // Create a temporary canvas for capture (not the display canvas)
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = this.video.videoWidth;
            captureCanvas.height = this.video.videoHeight;
            const captureCtx = captureCanvas.getContext('2d');
            
            // Draw current video frame
            captureCtx.drawImage(this.video, 0, 0, captureCanvas.width, captureCanvas.height);
            
            // Convert canvas to blob
            const blob = await new Promise(resolve => {
                captureCanvas.toBlob(resolve, 'image/jpeg', 0.8);
            });
            
            // Send to backend
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');
            
            const response = await fetch('/api/analyze-face', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.handleAnalysisResult(result);
            } else {
                console.warn('Frame analysis failed:', response.statusText);
            }
        } catch (error) {
            console.error('Capture error:', error);
        }
    }
    
    /**
     * Handle analysis result from backend
     */
    handleAnalysisResult(result) {
        if (result.success) {
            this.frameCount++;
            this.facialScores.push(result.scores);
            
            // Store face mesh data if provided
            if (result.face_mesh) {
                this.lastFaceMesh = result.face_mesh;
            }
            
            // Update UI in the widget
            document.getElementById('pain-score').textContent = result.scores.pain_score.toFixed(1);
            document.getElementById('stress-score').textContent = result.scores.stress_score.toFixed(1);
            document.getElementById('anxiety-score').textContent = result.scores.anxiety_score.toFixed(1);
            document.getElementById('frame-count').textContent = this.frameCount;
            
            // Update score colors based on severity
            this.updateScoreColors(result.scores);
            
            // CRITICAL: Update sessionStorage in REAL-TIME for voice page status box
            this.updateSessionStorageRealTime();
        }
    }
    
    /**
     * Update sessionStorage in real-time (called after every frame)
     * This allows the voice page status box to stay live and dynamic
     */
    updateSessionStorageRealTime() {
        // Calculate current averages
        const avgScores = this.getAverageScores();
        
        if (avgScores) {
            // Update the live analysis data
            sessionStorage.setItem('facial_analysis', JSON.stringify(avgScores));
            
            // Also update the accumulator
            const dataToStore = {
                all_frames: this.facialScores,
                total_frames: this.facialScores.length,
                timestamp: new Date().toISOString()
            };
            sessionStorage.setItem('facial_analysis_accumulator', JSON.stringify(dataToStore));
        }
    }
    
    /**
     * Update score display colors based on values
     */
    updateScoreColors(scores) {
        const updateColor = (elementId, value) => {
            const element = document.getElementById(elementId);
            if (value < 3) {
                element.style.color = '#10b981'; // Green
            } else if (value < 6) {
                element.style.color = '#f59e0b'; // Orange
            } else {
                element.style.color = '#ef4444'; // Red
            }
        };
        
        updateColor('pain-score', scores.pain_score);
        updateColor('stress-score', scores.stress_score);
        updateColor('anxiety-score', scores.anxiety_score);
    }
    
    /**
     * Update status text
     */
    updateStatus(text, type = '') {
        this.statusText.textContent = text;
        this.statusText.className = `status-badge ${type}`;
    }
    
    /**
     * Get average scores from all captured frames
     * CRITICAL: Uses ALL accumulated frames from all pages
     */
    getAverageScores() {
        if (this.facialScores.length === 0) {
            console.warn('⚠️ No facial scores available for averaging');
            return null;
        }
        
        const sum = this.facialScores.reduce((acc, scores) => ({
            pain_score: acc.pain_score + scores.pain_score,
            stress_score: acc.stress_score + scores.stress_score,
            anxiety_score: acc.anxiety_score + scores.anxiety_score
        }), { pain_score: 0, stress_score: 0, anxiety_score: 0 });
        
        const count = this.facialScores.length;
        
        const avgScores = {
            avg_pain_score: sum.pain_score / count,
            avg_stress_score: sum.stress_score / count,
            avg_anxiety_score: sum.anxiety_score / count,
            frame_count: count
        };
        
        console.log(`📊 Calculated averages from ${count} total frames:`, avgScores);
        
        return avgScores;
    }
    
    /**
     * Stop capture and cleanup
     */
    stop() {
        this.isCapturing = false;
        
        if (this.captureInterval) {
            clearInterval(this.captureInterval);
            this.captureInterval = null;
        }
        
        if (this.renderInterval) {
            clearInterval(this.renderInterval);
            this.renderInterval = null;
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.updateStatus('⏸️ Analysis stopped', '');
    }
    
    /**
     * Pause capture without stopping stream (for page transition)
     */
    pause() {
        this.isCapturing = false;
        
        if (this.captureInterval) {
            clearInterval(this.captureInterval);
            this.captureInterval = null;
        }
        
        if (this.renderInterval) {
            clearInterval(this.renderInterval);
            this.renderInterval = null;
        }
        
        // Mark that webcam is active for next page
        sessionStorage.setItem('webcam_active', 'true');
        
        this.updateStatus('⏸️ Transitioning...', '');
        console.log('🎥 Webcam stream kept active for voice page');
    }
    
    /**
     * Store facial data in session storage for next page
     * CRITICAL: Store ALL individual frames, not just averages
     */
    storeFacialData() {
        if (this.facialScores.length === 0) {
            console.warn('⚠️ No facial frames to store');
            return;
        }
        
        // Store ALL frames for accurate accumulation across pages
        const dataToStore = {
            all_frames: this.facialScores,  // Array of all frame scores
            total_frames: this.facialScores.length,
            timestamp: new Date().toISOString()
        };
        
        sessionStorage.setItem('facial_analysis_accumulator', JSON.stringify(dataToStore));
        console.log(`✅ Stored ${this.facialScores.length} frames in session storage`);
        
        // Also calculate and store average for backwards compatibility
        const avgScores = this.getAverageScores();
        if (avgScores) {
            sessionStorage.setItem('facial_analysis', JSON.stringify(avgScores));
        }
    }
}

// Global instance
let facialCapture = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🎭 DOMContentLoaded fired, pathname:', window.location.pathname);
    
    // Initialize on BOTH assessment and voice-analysis pages
    const isAssessmentPage = window.location.pathname.includes('/assessment') || 
                             document.querySelector('.assessment-form');
    const isVoicePage = window.location.pathname.includes('/voice-analysis');
    
    console.log('🎭 Page detection - assessment:', isAssessmentPage, 'voice:', isVoicePage);
    
    if (isAssessmentPage || isVoicePage) {
        // CRITICAL: Clear old data if starting fresh on assessment page
        if (isAssessmentPage) {
            console.log('🧹 Clearing previous facial data (starting fresh on assessment page)');
            sessionStorage.removeItem('facial_analysis_accumulator');
            sessionStorage.removeItem('facial_analysis');
        }
        
        console.log('🎭 Creating FacialCapture instance...');
        facialCapture = new FacialCapture();
        
        console.log('🎭 Calling init()...');
        await facialCapture.init();
        
        console.log(`✅ Facial capture initialized on ${isVoicePage ? 'voice' : 'assessment'} page`);
        
        // Intercept form submission to store facial data
        const form = document.querySelector('form');
        if (form) {
            console.log('🎭 Form found, adding submit listener');
            form.addEventListener('submit', (e) => {
                if (facialCapture) {
                    facialCapture.storeFacialData();
                    
                    // Only stop if going to results page, not voice page
                    if (isVoicePage) {
                        facialCapture.stop();
                        console.log('🛑 Facial capture stopped - going to results');
                    } else {
                        // Keep running for voice page
                        console.log('➡️ Facial capture continuing to voice page');
                    }
                }
            });
        } else {
            console.log('⚠️ No form found on page');
        }
    } else {
        console.log('ℹ️ Not on assessment or voice page, skipping facial capture');
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (facialCapture) {
        facialCapture.stop();
    }
});
