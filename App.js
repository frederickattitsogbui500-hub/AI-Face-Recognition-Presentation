// API Endpoint (Change to your Flask/FastAPI backend URL)
const API_URL = "http://localhost:5000/api/recognize";

// DOM Elements
const imageInput = document.getElementById('imageInput');
const fileName = document.getElementById('fileName');
const startCamBtn = document.getElementById('startCamBtn');
const captureBtn = document.getElementById('captureBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const imagePreview = document.getElementById('imagePreview');
const placeholderText = document.getElementById('placeholderText');
const resultBox = document.getElementById('resultBox');
const statusBadge = document.getElementById('statusBadge');
const resName = document.getElementById('resName');
const resConfidence = document.getElementById('resConfidence');
const resBox = document.getElementById('resBox');

let selectedBlob = null;
let stream = null;

// Handle File Selection
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedBlob = file;
        fileName.textContent = file.name;
        displayPreview(URL.createObjectURL(file));
        analyzeBtn.disabled = false;
    }
});

// Enable Webcam
startCamBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcam.srcObject = stream;
        captureBtn.disabled = false;
        startCamBtn.disabled = true;
    } catch (err) {
        alert("Camera access denied or unavailable.");
    }
});

// Capture Snapshot from Webcam
captureBtn.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    context.drawImage(webcam, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
        selectedBlob = blob;
        fileName.textContent = "Webcam_Snapshot.jpg";
        displayPreview(URL.createObjectURL(blob));
        analyzeBtn.disabled = false;
    }, 'image/jpeg');
});

// Show Image Preview
function displayPreview(url) {
    imagePreview.src = url;
    imagePreview.hidden = false;
    placeholderText.hidden = true;
    resultBox.classList.add('hidden');
}

// Send Image to Backend API
analyzeBtn.addEventListener('click', async () => {
    if (!selectedBlob) return;

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Processing...";

    const formData = new FormData();
    formData.append('image', selectedBlob, 'upload.jpg');

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Server error during recognition");

        const data = await response.json();
        updateUI(data);

    } catch (error) {
        alert("Failed to recognize face: " + error.message);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze Face";
    }
});

// Display API Response Results
function updateUI(data) {
    resultBox.classList.remove('hidden');

    if (data.matched) {
        statusBadge.textContent = "MATCH VERIFIED";
        statusBadge.className = "status-badge verified";
        resName.textContent = data.person_name;
        resConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        resBox.textContent = `[${data.bounding_box.join(', ')}]`;
    } else {
        statusBadge.textContent = "UNRECOGNIZED";
        statusBadge.className = "status-badge unknown";
        resName.textContent = "Unknown Individual";
        resConfidence.textContent = "N/A";
        resBox.textContent = data.bounding_box ? `[${data.bounding_box.join(', ')}]` : "None";
    }
}
