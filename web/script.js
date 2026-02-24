const API_URL = 'http://127.0.0.1:8000';

const uploadZone = document.getElementById('upload-zone');
const imageUpload = document.getElementById('image-upload');
const canvas = document.getElementById('visualizer-canvas');
const ctx = canvas.getContext('2d');
const resultsList = document.getElementById('results-list');
const detectionCount = document.getElementById('detection-count');
const statusDot = document.querySelector('.status-dot');
const apiStatus = document.getElementById('api-status');
const latencyDisplay = document.getElementById('latency-display');
const placeholder = document.getElementById('viewer-placeholder');

const tabs = document.querySelectorAll('.tab');
const controlSections = document.querySelectorAll('.control-section');

// Views handling
const navBtns = document.querySelectorAll('.nav-btn');
const viewPanels = document.querySelectorAll('.view-panel');

navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Update active nav button
        navBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update active view
        const view = btn.dataset.view;
        viewPanels.forEach(p => p.classList.remove('active-view'));
        document.getElementById(`view-${view}`).classList.add('active-view');

        // Hide inference controls if not on inference view
        const infControls = document.getElementById('inference-controls');
        if (view === 'analytics') {
            infControls.style.display = 'none';
        } else {
            infControls.style.display = 'flex';
        }
    });
});

window.refreshCharts = function () {
    const metrics = document.getElementById('chart-metrics');
    const cm = document.getElementById('chart-cm');
    const timestamp = new Date().getTime();

    if (metrics && metrics.tagName === 'IMG') {
        metrics.src = `${API_URL}/charts/training_metrics.png?t=${timestamp}`;
    }
    if (cm && cm.tagName === 'IMG') {
        cm.src = `${API_URL}/charts/confusion_matrix.png?t=${timestamp}`;
    }
};

// Camera elements
const video = document.getElementById('camera-video');
const startCameraBtn = document.getElementById('start-camera');
const stopCameraBtn = document.getElementById('stop-camera');
let stream = null;
let isStreaming = false;
let inferenceInterval = null;

// Tab Switching (Image vs Camera)
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const target = tab.dataset.source;
        controlSections.forEach(sec => sec.classList.remove('active-control'));
        document.getElementById(`${target}-controls`).classList.add('active-control');

        // Cleanup based on tab
        if (target === 'image' && isStreaming) {
            stopCamera();
        }
    });
});

// Setup File Upload
uploadZone.addEventListener('click', () => imageUpload.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--accent)';
    uploadZone.style.transform = 'translateY(-2px)';
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.style.borderColor = 'rgba(156, 163, 175, 0.3)';
    uploadZone.style.transform = 'translateY(0)';
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'rgba(156, 163, 175, 0.3)';
    uploadZone.style.transform = 'translateY(0)';
    if (e.dataTransfer.files.length) {
        processImageFile(e.dataTransfer.files[0]);
    }
});

imageUpload.addEventListener('change', (e) => {
    if (e.target.files.length) {
        processImageFile(e.target.files[0]);
    }
});

function drawImageToCanvas(img) {
    // Keep aspect ratio
    const maxWidth = canvas.parentElement.clientWidth;
    const maxHeight = canvas.parentElement.clientHeight;

    let w = img.width;
    let h = img.height;

    // Scale down if larger
    if (w > maxWidth) {
        h = h * (maxWidth / w);
        w = maxWidth;
    }
    if (h > maxHeight) {
        w = w * (maxHeight / h);
        h = maxHeight;
    }

    canvas.width = w;
    canvas.height = h;
    ctx.drawImage(img, 0, 0, w, h);

    return { w, h, origW: img.width, origH: img.height };
}

function processImageFile(file) {
    if (!file.type.startsWith('image/')) return;

    placeholder.style.display = 'none';
    canvas.style.display = 'block';
    video.style.display = 'none';

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
        const dims = drawImageToCanvas(img);
        sendToAPI(file, img, dims);
        URL.revokeObjectURL(url);
    };
    img.src = url;
}

// Draw bounding boxes on canvas
function drawDetections(plates, img, dims) {
    // Redraw base image
    ctx.drawImage(img, 0, 0, dims.w, dims.h);

    // Calculate scale factor
    const scaleX = dims.w / dims.origW;
    const scaleY = dims.h / dims.origH;

    plates.forEach(plate => {
        const [x1, y1, x2, y2] = plate.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        const scaledX = x1 * scaleX;
        const scaledY = y1 * scaleY;
        const scaledW = width * scaleX;
        const scaledH = height * scaleY;

        // Draw Box
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

        // Add a subtle glow to the box
        ctx.shadowColor = '#3b82f6';
        ctx.shadowBlur = 10;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
        ctx.shadowBlur = 0; // Reset

        // Draw Label bg
        ctx.fillStyle = 'rgba(17, 24, 39, 0.85)';
        ctx.fillRect(scaledX, scaledY - 30, scaledW, 30);

        // Draw Text
        ctx.fillStyle = '#fbbf24';
        ctx.font = 'bold 15px Outfit, monospace';
        ctx.fillText(plate.text || 'UNKNOWN', scaledX + 8, scaledY - 10);
    });
}

function updateResultsSidebar(plates) {
    detectionCount.innerText = plates.length;

    if (plates.length === 0) {
        resultsList.innerHTML = `
            <div class="empty-state">
                <i class="ph ph-car"></i>
                <p>No vehicles detected yet</p>
            </div>
        `;
        return;
    }

    resultsList.innerHTML = '';

    plates.forEach(plate => {
        const conf = Math.round(plate.confidence * 100);
        const html = `
            <div class="detection-card">
                <div class="detection-details">
                    <div class="detail-row">
                        <span class="detail-label"><i class="ph ph-hash"></i> Registration</span>
                        <span class="plate-text">${plate.text || 'UNREADABLE'}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label"><i class="ph ph-map-pin"></i> Region</span>
                        <span class="state-text">${plate.state}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label"><i class="ph ph-target"></i> Model Cond.</span>
                        <span class="conf-text">${conf}%</span>
                    </div>
                </div>
            </div>
        `;
        resultsList.insertAdjacentHTML('beforeend', html);
    });
}

async function sendToAPI(blob, originalImage, dims) {
    setProcessing(true);
    const start = performance.now();

    const formData = new FormData();
    formData.append('file', blob, 'capture.jpg');

    try {
        const res = await fetch(`${API_URL}/detect`, {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        const end = performance.now();
        latencyDisplay.innerHTML = `<i class="ph ph-clock"></i> ${Math.round(end - start)} ms`;

        if (data.success) {
            drawDetections(data.plates, originalImage ? originalImage : video, dims);
            updateResultsSidebar(data.plates);
        } else {
            console.error(data.error);
        }
    } catch (e) {
        console.error('API Error:', e);
        apiStatus.innerText = 'System Offline';
        statusDot.className = 'status-dot';
        statusDot.style.background = 'var(--danger)';
    } finally {
        setProcessing(false);
    }
}

function setProcessing(isProcessing) {
    if (isProcessing) {
        statusDot.className = 'status-dot processing';
        apiStatus.innerText = 'Inferencing...';
    } else {
        statusDot.className = 'status-dot online';
        apiStatus.innerText = 'System Online';
    }
}

// ---- Camera Processing ----

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        video.style.display = 'none'; // Hidden, drawn to canvas instead
        placeholder.style.display = 'none';
        canvas.style.display = 'block';

        startCameraBtn.disabled = true;
        stopCameraBtn.disabled = false;
        isStreaming = true;

        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            // Frame capture loop
            inferenceInterval = setInterval(captureAndSend, 1000); // 1 FPS for server load
        };
    } catch (e) {
        console.error("Camera access denied", e);
        alert("Camera access denied or unavailable.");
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    clearInterval(inferenceInterval);
    isStreaming = false;
    startCameraBtn.disabled = false;
    stopCameraBtn.disabled = true;
    placeholder.style.display = 'flex';
    canvas.style.display = 'none';
}

function captureAndSend() {
    if (!isStreaming) return;

    // Draw current video frame to a helper canvas to get BLOB
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);

    // Draw to main canvas for user viewing immediately
    const dims = drawImageToCanvas(video);

    tempCanvas.toBlob(blob => {
        sendToAPI(blob, video, dims);
    }, 'image/jpeg', 0.8);
}

startCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);
