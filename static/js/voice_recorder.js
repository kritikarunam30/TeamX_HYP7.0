/**
 * Voice Recorder - Handles audio recording and analysis
 * Integrates with VoiceAgent backend for emotion classification
 */

class VoiceRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.stream = null;
        this.analysisResult = null;
    }

    /**
     * Check if browser supports audio recording
     */
    isSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    /**
     * Request microphone permission and initialize
     */
    async initialize() {
        if (!this.isSupported()) {
            throw new Error('Audio recording not supported in this browser');
        }

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000  // 16kHz for optimal processing
                } 
            });
            console.log('✓ Microphone access granted');
            return true;
        } catch (error) {
            console.error('✗ Microphone access denied:', error);
            throw new Error('Microphone permission denied');
        }
    }

    /**
     * Start recording audio
     */
    startRecording() {
        if (!this.stream) {
            throw new Error('Microphone not initialized. Call initialize() first.');
        }

        this.audioChunks = [];
        
        // Create MediaRecorder with WAV format if possible
        const mimeType = this.getSupportedMimeType();
        this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });

        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.audioChunks.push(event.data);
            }
        };

        this.mediaRecorder.onstop = () => {
            console.log('Recording stopped');
        };

        this.mediaRecorder.start();
        this.isRecording = true;
        console.log(`✓ Recording started (${mimeType})`);
    }

    /**
     * Stop recording and get audio blob
     */
    stopRecording() {
        return new Promise((resolve, reject) => {
            if (!this.mediaRecorder || !this.isRecording) {
                reject(new Error('Not currently recording'));
                return;
            }

            this.mediaRecorder.onstop = () => {
                const mimeType = this.mediaRecorder.mimeType;
                const audioBlob = new Blob(this.audioChunks, { type: mimeType });
                this.isRecording = false;
                console.log(`✓ Recording completed: ${(audioBlob.size / 1024).toFixed(2)} KB`);
                resolve(audioBlob);
            };

            this.mediaRecorder.stop();
        });
    }

    /**
     * Get supported MIME type for recording
     */
    getSupportedMimeType() {
        const types = [
            'audio/webm',
            'audio/webm;codecs=opus',
            'audio/ogg;codecs=opus',
            'audio/mp4'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }

        return '';  // Let browser choose default
    }

    /**
     * Upload audio to server for analysis
     */
    async analyzeAudio(audioBlob) {
        const formData = new FormData();
        
        // Convert to WAV if possible for better compatibility
        const filename = `voice_recording_${Date.now()}.wav`;
        formData.append('audio', audioBlob, filename);

        try {
            const response = await fetch('/api/analyze-voice', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Analysis failed');
            }

            this.analysisResult = await response.json();
            console.log('✓ Voice analysis complete:', this.analysisResult);
            return this.analysisResult;

        } catch (error) {
            console.error('✗ Voice analysis error:', error);
            throw error;
        }
    }

    /**
     * Stop all tracks and release microphone
     */
    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            console.log('✓ Microphone released');
        }
    }

    /**
     * Get current analysis result
     */
    getAnalysisResult() {
        return this.analysisResult;
    }
}

/**
 * UI Controller for voice recording interface
 */
class VoiceRecorderUI {
    constructor(recorder) {
        this.recorder = recorder;
        this.recordButton = null;
        this.stopButton = null;
        this.analyzeButton = null;
        this.statusText = null;
        this.resultsContainer = null;
        this.recordingTime = 0;
        this.timerInterval = null;
    }

    /**
     * Initialize UI elements
     */
    init(elements) {
        this.recordButton = elements.recordButton;
        this.stopButton = elements.stopButton;
        this.analyzeButton = elements.analyzeButton;
        this.statusText = elements.statusText;
        this.resultsContainer = elements.resultsContainer;
        this.timerElement = elements.timerElement;

        this.setupEventListeners();
        this.updateUIState('idle');
    }

    /**
     * Setup button event listeners
     */
    setupEventListeners() {
        if (this.recordButton) {
            this.recordButton.addEventListener('click', () => this.handleRecord());
        }

        if (this.stopButton) {
            this.stopButton.addEventListener('click', () => this.handleStop());
        }

        if (this.analyzeButton) {
            this.analyzeButton.addEventListener('click', () => this.handleAnalyze());
        }
    }

    /**
     * Handle record button click
     */
    async handleRecord() {
        try {
            if (!this.recorder.stream) {
                this.updateStatus('Requesting microphone access...', 'info');
                await this.recorder.initialize();
            }

            this.recorder.startRecording();
            this.updateUIState('recording');
            this.startTimer();
            this.updateStatus('Recording... Speak naturally', 'recording');

        } catch (error) {
            this.updateStatus(`Error: ${error.message}`, 'error');
            this.updateUIState('idle');
        }
    }

    /**
     * Handle stop button click
     */
    async handleStop() {
        try {
            this.stopTimer();
            this.updateStatus('Processing recording...', 'info');
            
            const audioBlob = await this.recorder.stopRecording();
            
            // Save blob for analysis
            this.currentAudioBlob = audioBlob;
            
            this.updateUIState('stopped');
            this.updateStatus(`Recording saved (${this.formatTime(this.recordingTime)})`, 'success');

        } catch (error) {
            this.updateStatus(`Error: ${error.message}`, 'error');
            this.updateUIState('idle');
        }
    }

    /**
     * Handle analyze button click
     */
    async handleAnalyze() {
        if (!this.currentAudioBlob) {
            this.updateStatus('No recording to analyze', 'error');
            return;
        }

        try {
            this.updateUIState('analyzing');
            this.updateStatus('Analyzing voice patterns...', 'info');

            const result = await this.recorder.analyzeAudio(this.currentAudioBlob);
            
            // Store result in sessionStorage for form submission
            sessionStorage.setItem('voice_analysis', JSON.stringify(result));
            console.log('✅ Voice analysis stored in sessionStorage:', result);
            
            this.displayResults(result);
            this.updateUIState('analyzed');
            this.updateStatus('Analysis complete!', 'success');

        } catch (error) {
            this.updateStatus(`Analysis error: ${error.message}`, 'error');
            this.updateUIState('stopped');
        }
    }

    /**
     * Update UI state
     */
    updateUIState(state) {
        // Hide all buttons first
        if (this.recordButton) this.recordButton.style.display = 'none';
        if (this.stopButton) this.stopButton.style.display = 'none';
        if (this.analyzeButton) this.analyzeButton.style.display = 'none';

        switch (state) {
            case 'idle':
                if (this.recordButton) this.recordButton.style.display = 'inline-block';
                break;
            case 'recording':
                if (this.stopButton) this.stopButton.style.display = 'inline-block';
                break;
            case 'stopped':
                if (this.recordButton) this.recordButton.style.display = 'inline-block';
                if (this.analyzeButton) this.analyzeButton.style.display = 'inline-block';
                break;
            case 'analyzing':
                // Show loading state
                break;
            case 'analyzed':
                if (this.recordButton) this.recordButton.style.display = 'inline-block';
                break;
        }
    }

    /**
     * Update status text
     */
    updateStatus(message, type = 'info') {
        if (!this.statusText) return;

        this.statusText.textContent = message;
        
        // Update color based on type
        const colors = {
            'info': '#64748b',
            'success': '#10b981',
            'error': '#ef4444',
            'recording': '#f59e0b'
        };
        
        this.statusText.style.color = colors[type] || colors.info;
    }

    /**
     * Start recording timer
     */
    startTimer() {
        this.recordingTime = 0;
        this.updateTimer();
        
        this.timerInterval = setInterval(() => {
            this.recordingTime++;
            this.updateTimer();
        }, 1000);
    }

    /**
     * Stop recording timer
     */
    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    /**
     * Update timer display
     */
    updateTimer() {
        if (this.timerElement) {
            this.timerElement.textContent = this.formatTime(this.recordingTime);
        }
    }

    /**
     * Format seconds to MM:SS
     */
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    /**
     * Display analysis results
     */
    displayResults(result) {
        if (!this.resultsContainer) return;

        const html = `
            <div class="voice-results">
                <h3 style="color: #075985; margin-bottom: 20px;">🎤 Voice Analysis Results</h3>
                
                <div class="result-grid">
                    <div class="result-card">
                        <div class="result-label">Primary Emotion</div>
                        <div class="result-value">${this.capitalizeFirst(result.emotion_primary)}</div>
                    </div>
                    
                    <div class="result-card">
                        <div class="result-label">Secondary Emotion</div>
                        <div class="result-value">${this.capitalizeFirst(result.emotion_secondary)}</div>
                    </div>
                    
                    <div class="result-card">
                        <div class="result-label">Stress Level</div>
                        <div class="result-value">${result.stress_score}/10</div>
                        <div class="result-bar">
                            <div class="result-bar-fill" style="width: ${result.stress_score * 10}%; background: ${this.getScoreColor(result.stress_score)}"></div>
                        </div>
                    </div>
                    
                    <div class="result-card">
                        <div class="result-label">Fatigue Level</div>
                        <div class="result-value">${result.fatigue_score}/10</div>
                        <div class="result-bar">
                            <div class="result-bar-fill" style="width: ${result.fatigue_score * 10}%; background: ${this.getScoreColor(result.fatigue_score)}"></div>
                        </div>
                    </div>
                    
                    <div class="result-card">
                        <div class="result-label">Voice Quality</div>
                        <div class="result-value">${result.voice_quality_score}/10</div>
                        <div class="result-bar">
                            <div class="result-bar-fill" style="width: ${result.voice_quality_score * 10}%; background: ${this.getScoreColor(10 - result.voice_quality_score)}"></div>
                        </div>
                    </div>
                    
                    <div class="result-card">
                        <div class="result-label">Speaking Rate</div>
                        <div class="result-value">${result.speaking_rate} syl/sec</div>
                    </div>
                </div>
                
                <div class="result-explanation">
                    <strong>Analysis:</strong> ${result.explanation}
                </div>
            </div>
        `;

        this.resultsContainer.innerHTML = html;
        this.resultsContainer.style.display = 'block';
    }

    /**
     * Get color based on score (0-10, higher = worse)
     */
    getScoreColor(score) {
        if (score < 3) return '#10b981';  // Green
        if (score < 7) return '#f59e0b';  // Orange
        return '#ef4444';  // Red
    }

    /**
     * Capitalize first letter
     */
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

// Export for use in HTML
window.VoiceRecorder = VoiceRecorder;
window.VoiceRecorderUI = VoiceRecorderUI;
