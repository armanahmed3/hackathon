// Global variables
let classes = [];
let selectedImage = null;
let trainingTaskId = null;
let trainingWebSocket = null;
let lossChart = null;
let accuracyChart = null;

// DOM Elements
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

// Dataset tab elements
const classNameInput = document.getElementById('class-name');
const createClassBtn = document.getElementById('create-class-btn');
const classMessage = document.getElementById('class-message');
const selectClass = document.getElementById('select-class');
const imageFileInput = document.getElementById('image-file');
const uploadImageBtn = document.getElementById('upload-image-btn');
const uploadMessage = document.getElementById('upload-message');
const webcamClass = document.getElementById('webcam-class');
const startCameraBtn = document.getElementById('start-camera-btn');
const captureBtn = document.getElementById('capture-btn');
const webcamMessage = document.getElementById('webcam-message');
const datasetPreview = document.getElementById('dataset-preview');

// Training tab elements
const modelCheckboxes = document.querySelectorAll('.model-checkbox');
const classSelection = document.getElementById('class-selection');
const startTrainingBtn = document.getElementById('start-training-btn');
const trainingMessage = document.getElementById('training-message');
const trainingStatus = document.getElementById('training-status');
const progressBar = document.getElementById('progress-bar');
const progressPercent = document.getElementById('progress-percent');
const lossCanvas = document.getElementById('loss-chart');
const accuracyCanvas = document.getElementById('accuracy-chart');

// Evaluation tab elements
const evalModelSelect = document.getElementById('eval-model-select');
const evaluateBtn = document.getElementById('evaluate-btn');
const evaluationResults = document.getElementById('evaluation-results');

// Prediction tab elements
const uploadPredictBtn = document.getElementById('upload-predict-btn');
const webcamPredictBtn = document.getElementById('webcam-predict-btn');
const predictImageFile = document.getElementById('predict-image-file');
const predictionWebcamContainer = document.getElementById('prediction-webcam-container');
const predictionWebcam = document.getElementById('prediction-webcam');
const startPredictionWebcamBtn = document.getElementById('start-prediction-webcam-btn');
const capturePredictionBtn = document.getElementById('capture-prediction-btn');
const previewImage = document.getElementById('preview-image');
const predictBtn = document.getElementById('predict-btn');
const predictionResults = document.getElementById('prediction-results');
const predictionMessage = document.getElementById('prediction-message');

// API Base URL
const API_BASE_URL = 'http://localhost:8001';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeTabs();
    initializeWebcam();
    initializeCharts();
    loadClasses();
    
    // Add animation to cards when they appear
    setTimeout(() => {
        document.querySelectorAll('.card').forEach((card, index) => {
            setTimeout(() => {
                card.classList.add('fade-in');
            }, index * 100);
        });
    }, 300);
});

// Tab navigation
function initializeTabs() {
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.getAttribute('data-tab');
            
            // Update active tab button
            tabButtons.forEach(btn => {
                if (btn === button) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
            
            // Show active tab content
            tabContents.forEach(content => {
                if (content.id === `${tabName}-tab`) {
                    content.classList.remove('hidden');
                } else {
                    content.classList.add('hidden');
                }
            });
        });
    });
}

// Initialize webcam
function initializeWebcam() {
    // Configure Webcam.js
    Webcam.set({
        width: 320,
        height: 240,
        image_format: 'jpeg',
        jpeg_quality: 90
    });
    
    // Event listeners for dataset webcam
    startCameraBtn.addEventListener('click', startWebcam);
    captureBtn.addEventListener('click', captureImage);
    
    // Event listeners for prediction webcam
    uploadPredictBtn.addEventListener('click', showUploadOption);
    webcamPredictBtn.addEventListener('click', showWebcamOption);
    startPredictionWebcamBtn.addEventListener('click', startPredictionWebcam);
    capturePredictionBtn.addEventListener('click', captureAndPredict);
    
    // Event listener for image upload
    predictImageFile.addEventListener('change', handleImageUpload);
}

// Initialize charts
function initializeCharts() {
    const lossCtx = lossCanvas.getContext('2d');
    const accuracyCtx = accuracyCanvas.getContext('2d');
    
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.1,
                fill: true
            }, {
                label: 'Validation Loss',
                data: [],
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
    
    accuracyChart = new Chart(accuracyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Accuracy',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
                fill: true
            }, {
                label: 'Validation Accuracy',
                data: [],
                borderColor: 'rgb(153, 102, 255)',
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
}

// Load classes from backend
async function loadClasses() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/classes`);
        const data = await response.json();
        classes = data.classes || [];
        updateClassDropdowns();
        updateDatasetPreview();
        updateClassSelection();
    } catch (error) {
        console.error('Error loading classes:', error);
    }
}

// Update class dropdowns
function updateClassDropdowns() {
    // Clear existing options
    selectClass.innerHTML = '<option value="">Select a class</option>';
    webcamClass.innerHTML = '<option value="">Select a class</option>';
    
    // Add classes to dropdowns
    classes.forEach(cls => {
        const option1 = document.createElement('option');
        option1.value = cls.name;
        option1.textContent = `${cls.name} (${cls.count} images)`;
        selectClass.appendChild(option1);
        
        const option2 = document.createElement('option');
        option2.value = cls.name;
        option2.textContent = `${cls.name} (${cls.count} images)`;
        webcamClass.appendChild(option2);
    });
}

// Update dataset preview
function updateDatasetPreview() {
    if (classes.length === 0) {
        datasetPreview.innerHTML = '<p class="text-gray-500 text-center py-8">No classes created yet.</p>';
        return;
    }
    
    let html = '';
    classes.forEach(cls => {
        html += `
            <div class="class-card">
                <span class="class-name">${cls.name}</span>
                <span class="class-count">${cls.count} images</span>
                <button class="delete-class-btn" data-class="${cls.name}">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
    });
    datasetPreview.innerHTML = html;
    
    // Add event listeners to delete buttons
    document.querySelectorAll('.delete-class-btn').forEach(button => {
        button.addEventListener('click', async (e) => {
            const className = e.target.closest('.delete-class-btn').dataset.class;
            if (confirm(`Are you sure you want to delete the class "${className}" and all its images?`)) {
                try {
                    const response = await fetch(`${API_BASE_URL}/api/classes/${className}`, {
                        method: 'DELETE'
                    });
                    
                    if (response.ok) {
                        showMessage(classMessage, `Class "${className}" deleted successfully!`, 'success');
                        loadClasses();
                    } else {
                        const error = await response.json();
                        showMessage(classMessage, error.detail || 'Failed to delete class.', 'error');
                    }
                } catch (error) {
                    showMessage(classMessage, 'Error deleting class: ' + error.message, 'error');
                }
            }
        });
    });
}

// Update class selection checkboxes
function updateClassSelection() {
    if (classes.length === 0) {
        classSelection.innerHTML = '<p class="text-gray-500">No classes available. Create classes in the Dataset tab.</p>';
        startTrainingBtn.disabled = true;
        return;
    }
    
    let html = '';
    classes.forEach(cls => {
        html += `
            <div class="flex items-center p-2 hover:bg-gray-50 rounded-lg">
                <input type="checkbox" id="class-${cls.name}" class="class-checkbox h-5 w-5 text-blue-600 rounded" value="${cls.name}">
                <label for="class-${cls.name}" class="ml-2 text-gray-700 font-medium">${cls.name} <span class="text-gray-500">(${cls.count} images)</span></label>
            </div>
        `;
    });
    classSelection.innerHTML = html;
    startTrainingBtn.disabled = false;
    
    // Add event listeners to class checkboxes
    document.querySelectorAll('.class-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', checkTrainingRequirements);
    });
}

// Check if training can be started
function checkTrainingRequirements() {
    const selectedModels = Array.from(modelCheckboxes).filter(cb => cb.checked).length;
    const selectedClasses = Array.from(document.querySelectorAll('.class-checkbox')).filter(cb => cb.checked).length;
    
    startTrainingBtn.disabled = !(selectedModels > 0 && selectedClasses > 0);
}

// Create a new class
createClassBtn.addEventListener('click', async () => {
    const className = classNameInput.value.trim();
    if (!className) {
        showMessage(classMessage, 'Please enter a class name.', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/classes`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ class_name: className })
        });
        
        if (response.ok) {
            showMessage(classMessage, `Class "${className}" created successfully!`, 'success');
            classNameInput.value = '';
            loadClasses();
        } else {
            const error = await response.json();
            showMessage(classMessage, error.detail || 'Failed to create class.', 'error');
        }
    } catch (error) {
        showMessage(classMessage, 'Error creating class: ' + error.message, 'error');
    }
});

// Upload image
uploadImageBtn.addEventListener('click', async () => {
    const className = selectClass.value;
    const files = imageFileInput.files;
    
    if (!className) {
        showMessage(uploadMessage, 'Please select a class.', 'error');
        return;
    }
    
    if (files.length === 0) {
        showMessage(uploadMessage, 'Please select at least one image file.', 'error');
        return;
    }
    
    // Upload each file
    let successCount = 0;
    let errorCount = 0;
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append('class_name', className);
        formData.append('file', file);
        
        try {
            const response = await fetch(`${API_BASE_URL}/api/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                successCount++;
            } else {
                errorCount++;
                const error = await response.json();
                console.error(`Error uploading ${file.name}:`, error.detail || 'Failed to upload image.');
            }
        } catch (error) {
            errorCount++;
            console.error(`Error uploading ${file.name}:`, error.message);
        }
    }
    
    // Show result message
    if (errorCount === 0) {
        showMessage(uploadMessage, `Successfully uploaded ${successCount} image(s)!`, 'success');
    } else if (successCount > 0) {
        showMessage(uploadMessage, `Uploaded ${successCount} image(s) successfully, ${errorCount} failed.`, 'success');
    } else {
        showMessage(uploadMessage, `Failed to upload ${errorCount} image(s).`, 'error');
    }
    
    imageFileInput.value = '';
    loadClasses(); // Refresh class counts
});

// Start webcam for dataset
async function startWebcam() {
    const className = webcamClass.value;
    if (!className) {
        showMessage(webcamMessage, 'Please select a class.', 'error');
        return;
    }
    
    try {
        await Webcam.attach('#webcam');
        captureBtn.disabled = false;
        showMessage(webcamMessage, 'Webcam started. Click Capture to take a photo.', 'success');
    } catch (error) {
        showMessage(webcamMessage, 'Error starting webcam: ' + error.message, 'error');
    }
}

// Capture image from webcam
function captureImage() {
    const className = webcamClass.value;
    if (!className) {
        showMessage(webcamMessage, 'Please select a class.', 'error');
        return;
    }
    
    // Show capture overlay animation
    const overlay = document.getElementById('capture-overlay');
    overlay.classList.add('active');
    setTimeout(() => {
        overlay.classList.remove('active');
    }, 300);
    
    Webcam.snap(async (data_uri) => {
        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('class_name', className);
            formData.append('image_base64', data_uri);
            
            // Send to backend
            const response = await fetch(`${API_BASE_URL}/api/capture`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                showMessage(webcamMessage, 'Image captured and saved successfully!', 'success');
                loadClasses(); // Refresh class counts
            } else {
                const error = await response.json();
                showMessage(webcamMessage, error.detail || 'Failed to save captured image.', 'error');
            }
        } catch (error) {
            showMessage(webcamMessage, 'Error saving captured image: ' + error.message, 'error');
        }
    });
}

// Start training
startTrainingBtn.addEventListener('click', async () => {
    const selectedModels = Array.from(modelCheckboxes).filter(cb => cb.checked).map(cb => cb.value);
    const selectedClasses = Array.from(document.querySelectorAll('.class-checkbox')).filter(cb => cb.checked).map(cb => cb.value);
    
    if (selectedModels.length === 0) {
        showMessage(trainingMessage, 'Please select at least one model.', 'error');
        return;
    }
    
    if (selectedClasses.length === 0) {
        showMessage(trainingMessage, 'Please select at least one class.', 'error');
        return;
    }
    
    // Check if each selected class has at least 10 images
    const classWithInsufficientImages = classes.filter(cls => 
        selectedClasses.includes(cls.name) && cls.count < 10
    );
    
    if (classWithInsufficientImages.length > 0) {
        const classNames = classWithInsufficientImages.map(cls => cls.name).join(', ');
        showMessage(trainingMessage, `Classes ${classNames} have less than 10 images. Please add more images.`, 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                models: selectedModels,
                classes: selectedClasses
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            trainingTaskId = data.task_id;
            
            showMessage(trainingMessage, 'Training started successfully!', 'success');
            startTrainingBtn.disabled = true;
            
            // Connect to WebSocket for real-time updates
            connectToTrainingWebSocket(trainingTaskId);
            
            // Update UI
            trainingStatus.innerHTML = '<p class="text-blue-600 text-center py-4">Training in progress... <i class="fas fa-spinner fa-spin"></i></p>';
            resetCharts();
        } else {
            const error = await response.json();
            showMessage(trainingMessage, error.detail || 'Failed to start training.', 'error');
        }
    } catch (error) {
        showMessage(trainingMessage, 'Error starting training: ' + error.message, 'error');
    }
});

// Connect to training WebSocket
function connectToTrainingWebSocket(taskId) {
    if (trainingWebSocket) {
        trainingWebSocket.close();
    }
    
    trainingWebSocket = new WebSocket(`ws://localhost:8001/ws/train/${taskId}`);
    
    trainingWebSocket.onopen = function(event) {
        console.log('Connected to training WebSocket');
    };
    
    trainingWebSocket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        // Update progress bar
        const progress = data.progress || 0;
        progressBar.style.width = `${progress}%`;
        progressPercent.textContent = `${Math.round(progress)}%`;
        
        // Update status message
        if (data.status === 'completed') {
            trainingStatus.innerHTML = '<p class="text-green-600 text-center py-4">Training completed! <i class="fas fa-check-circle"></i></p>';
            startTrainingBtn.disabled = false;
            if (trainingWebSocket) {
                trainingWebSocket.close();
            }
        } else if (data.status === 'error') {
            trainingStatus.innerHTML = `<p class="text-red-600 text-center py-4">Training error: ${data.error || 'Unknown error'} <i class="fas fa-exclamation-triangle"></i></p>`;
            startTrainingBtn.disabled = false;
            if (trainingWebSocket) {
                trainingWebSocket.close();
            }
        } else {
            trainingStatus.innerHTML = `<p class="text-blue-600 text-center py-4">Training in progress... (${Math.round(progress)}%) <i class="fas fa-spinner fa-spin"></i></p>`;
        }
        
        // Update charts if training data is available
        if (data.results) {
            updateCharts(data.results);
        }
    };
    
    trainingWebSocket.onerror = function(error) {
        console.error('WebSocket error:', error);
        trainingStatus.innerHTML = '<p class="text-red-600 text-center py-4">WebSocket connection error. <i class="fas fa-exclamation-triangle"></i></p>';
    };
    
    trainingWebSocket.onclose = function(event) {
        console.log('WebSocket connection closed');
    };
}

// Reset charts
function resetCharts() {
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.data.datasets[1].data = [];
    lossChart.update();
    
    accuracyChart.data.labels = [];
    accuracyChart.data.datasets[0].data = [];
    accuracyChart.data.datasets[1].data = [];
    accuracyChart.update();
}

// Update charts with training data
function updateCharts(results) {
    // For simplicity, we'll just show the final results
    // In a real implementation, you would update the charts with each epoch's data
    console.log('Training results:', results);
}

// Evaluate model
evaluateBtn.addEventListener('click', async () => {
    const modelName = evalModelSelect.value;
    if (!modelName) {
        showMessage(evaluationResults, 'Please select a model to evaluate.', 'error');
        return;
    }
    
    try {
        evaluationResults.innerHTML = '<p class="text-gray-500 text-center py-4">Evaluating model... <i class="fas fa-spinner fa-spin"></i></p>';
        
        const response = await fetch(`${API_BASE_URL}/api/evaluate/${modelName}`);
        
        if (response.ok) {
            const data = await response.json();
            
            // Display results
            let html = `
                <div class="prediction-card mb-4">
                    <h3 class="prediction-model ${modelName.replace('_', '-')}">
                        <i class="fas fa-chart-line mr-2"></i>Evaluation Results: ${modelName.replace('_', ' ').toUpperCase()}
                    </h3>
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-blue-50 p-3 rounded-lg">
                            <p class="text-gray-700 font-medium">Accuracy</p>
                            <p class="text-2xl font-bold text-blue-600">${(data.accuracy * 100).toFixed(2)}%</p>
                        </div>
                    </div>
                </div>
            `;
            
            // Confusion matrix (simplified representation)
            if (data.confusion_matrix) {
                html += `
                    <div class="prediction-card">
                        <h3 class="prediction-model">
                            <i class="fas fa-th mr-2"></i>Confusion Matrix
                        </h3>
                        <div class="overflow-x-auto">
                            <table class="min-w-full divide-y divide-gray-200">
                                <thead class="bg-gray-50">
                                    <tr>
                                        <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actual/Predicted</th>
                                        ${classes.map(cls => `<th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">${cls.name}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody class="bg-white divide-y divide-gray-200">
                                    ${data.confusion_matrix.map((row, i) => `
                                        <tr>
                                            <td class="px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-900">${classes[i]?.name || `Class ${i}`}</td>
                                            ${row.map(val => `<td class="px-4 py-2 whitespace-nowrap text-sm text-gray-500">${val}</td>`).join('')}
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            }
            
            evaluationResults.innerHTML = html;
        } else {
            const error = await response.json();
            showMessage(evaluationResults, error.detail || 'Failed to evaluate model.', 'error');
        }
    } catch (error) {
        showMessage(evaluationResults, 'Error evaluating model: ' + error.message, 'error');
    }
});

// Show upload option for prediction
function showUploadOption() {
    predictImageFile.classList.remove('hidden');
    predictionWebcamContainer.classList.add('hidden');
    predictImageFile.click();
}

// Show webcam option for prediction
function showWebcamOption() {
    predictImageFile.classList.add('hidden');
    predictionWebcamContainer.classList.remove('hidden');
    previewImage.src = '';
    previewImage.classList.add('hidden');
    predictBtn.disabled = true;
}

// Handle image upload for prediction
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');
        selectedImage = file;
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Start webcam for prediction
async function startPredictionWebcam() {
    try {
        await Webcam.attach('#prediction-webcam');
        capturePredictionBtn.disabled = false;
        showMessage(predictionMessage, 'Webcam started. Click Capture & Predict to take a photo and make prediction.', 'success');
    } catch (error) {
        showMessage(predictionMessage, 'Error starting webcam: ' + error.message, 'error');
    }
}

// Capture image and make prediction
function captureAndPredict() {
    Webcam.snap(async (data_uri) => {
        try {
            // Display preview
            previewImage.src = data_uri;
            previewImage.classList.remove('hidden');
            
            // Convert data URI to blob for prediction
            const response = await fetch(data_uri);
            selectedImage = await response.blob();
            predictBtn.disabled = false;
            
            // Make prediction
            makePrediction();
        } catch (error) {
            showMessage(predictionMessage, 'Error capturing image: ' + error.message, 'error');
        }
    });
}

// Make prediction
predictBtn.addEventListener('click', makePrediction);

async function makePrediction() {
    if (!selectedImage) {
        showMessage(predictionMessage, 'Please select an image or capture from webcam.', 'error');
        return;
    }
    
    try {
        predictionResults.innerHTML = '<p class="text-gray-500 text-center py-8">Making prediction... <i class="fas fa-spinner fa-spin"></i></p>';
        
        const formData = new FormData();
        formData.append('file', selectedImage);
        
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // Display results
            let html = '';
            
            // Show predictions for each model
            if (data.logistic_regression) {
                html += `
                    <div class="prediction-card">
                        <h3 class="prediction-model logistic">
                            <i class="fas fa-calculator mr-2"></i>Logistic Regression
                        </h3>
                        <p class="prediction-value">${data.logistic_regression}</p>
                        ${renderProbabilities(data.probabilities?.logistic)}
                    </div>
                `;
            }
            
            if (data.random_forest) {
                html += `
                    <div class="prediction-card">
                        <h3 class="prediction-model random-forest">
                            <i class="fas fa-tree mr-2"></i>Random Forest
                        </h3>
                        <p class="prediction-value">${data.random_forest}</p>
                        ${renderProbabilities(data.probabilities?.random_forest)}
                    </div>
                `;
            }
            
            if (data.cnn) {
                html += `
                    <div class="prediction-card">
                        <h3 class="prediction-model cnn">
                            <i class="fas fa-network-wired mr-2"></i>CNN
                        </h3>
                        <p class="prediction-value">${data.cnn}</p>
                        ${renderProbabilities(data.probabilities?.cnn)}
                    </div>
                `;
            }
            
            predictionResults.innerHTML = html;
        } else {
            const error = await response.json();
            showMessage(predictionMessage, error.detail || 'Failed to make prediction.', 'error');
        }
    } catch (error) {
        showMessage(predictionMessage, 'Error making prediction: ' + error.message, 'error');
    }
}

// Render probabilities
function renderProbabilities(probabilities) {
    if (!probabilities || Object.keys(probabilities).length === 0) {
        return '<p class="text-sm text-gray-500 mt-2">No probability data available.</p>';
    }
    
    let html = '<div class="probabilities"><h4 class="font-medium mb-2">Probabilities:</h4><div class="space-y-1">';
    for (const [className, prob] of Object.entries(probabilities)) {
        const percentage = (prob * 100).toFixed(2);
        html += `
            <div class="probability-item">
                <span class="probability-label">${className}</span>
                <span class="probability-value">${percentage}%</span>
            </div>
        `;
    }
    html += '</div></div>';
    return html;
}

// Show message in UI
function showMessage(element, message, type) {
    const isError = type === 'error';
    const isSuccess = type === 'success';
    
    element.textContent = message;
    element.className = `notification ${type}`;
    element.style.display = 'flex';
    
    // Add icon based on type
    const icon = isError ? '⚠️' : isSuccess ? '✅' : 'ℹ️';
    element.innerHTML = `
        <span class="notification-icon">${icon}</span>
        <span>${message}</span>
    `;
    
    // Clear message after 5 seconds
    setTimeout(() => {
        element.style.display = 'none';
    }, 5000);
}

// Add event listeners to model checkboxes
modelCheckboxes.forEach(checkbox => {
    checkbox.addEventListener('change', checkTrainingRequirements);
});