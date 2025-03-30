let mobilenet;
let model;
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0, spockSamples=0, lizardSamples=0, customSamples=0;
let isPredicting = false;
let predictionHistory = [];
let boostMode = false;
let multiplayerMode = false;
let speedChallenge = false;
let speedStartTime, speedSequence = [];
let scores = { wins: 0, losses: 0, ties: 0 };
let leaderboard = JSON.parse(localStorage.getItem('rpslsLeaderboard')) || [];
let hands, canvas, ctx;
let trainingStats = { accuracy: 0, loss: 0 };
let videoStream;
let predictionThrottle = false;
let loadingElement;
let statusElement;
let lastPrediction = -1;
let predictionBuffer = [];
let frameCount = 0;


let worker = null;
if (window.Worker) {
    try {


    } catch (e) {
        console.log('Worker creation failed:', e);
    }
}


document.addEventListener('DOMContentLoaded', () => {
    loadingElement = document.getElementById('loading');
    statusElement = document.getElementById('status');
    updateStatus('initializing');
});

function updateStatus(state) {
    if (!statusElement) return;
    
    const indicator = statusElement.querySelector('.status-indicator');
    const states = {
        'initializing': { text: 'Initializing...', class: 'status-inactive' },
        'ready': { text: 'Ready', class: 'status-inactive' },
        'training': { text: 'Training Model...', class: 'status-active' },
        'predicting': { text: 'Predicting', class: 'status-active' },
        'loading': { text: 'Loading Model...', class: 'status-active' },
        'error': { text: 'Error', class: 'status-inactive' }
    };
    
    if (states[state]) {
        statusElement.innerHTML = `Status: <span class="status-indicator ${states[state].class}"></span> ${states[state].text}`;
    }
}

function showLoading(show, message = "Loading...") {
    if (loadingElement) {
        if (show) {

            let messageElement = loadingElement.querySelector('.loading-message');
            if (!messageElement) {
                messageElement = document.createElement('div');
                messageElement.className = 'loading-message';
                messageElement.style.color = '#00ffff';
                messageElement.style.marginTop = '15px';
                messageElement.style.fontFamily = "'Orbitron', sans-serif";
                messageElement.style.textAlign = 'center';
                loadingElement.appendChild(messageElement);
            }
            messageElement.textContent = message;
            loadingElement.style.display = 'flex';
            loadingElement.style.opacity = '1';
        } else {

            loadingElement.style.opacity = '0';
            setTimeout(() => {
                loadingElement.style.display = 'none';
            }, 300);
        }
    }
}

async function loadMobilenet() {
    updateStatus('initializing');
    showLoading(true);
    try {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
        showLoading(false);
        updateStatus('ready');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
    } catch (error) {
        console.error("Failed to load MobileNet:", error);
        showLoading(false);
        updateStatus('error');
        alert("Failed to load MobileNet. Check your internet connection and try again.");
        return null;
    }
}


function cleanup() {
    if (tf.memory().numTensors > 50) {
        console.log('Memory cleanup - tensors before:', tf.memory().numTensors);
        tf.tidy(() => {});
        console.log('Tensors after cleanup:', tf.memory().numTensors);
    }
}


async function train(epochs = 10) {
    if (dataset.labels.length < 5) {
        alert("Please add at least 5 samples of each gesture before training.");
        return;
    }
    
    updateStatus('training');
    showLoading(true);
    

    const feedbackElement = document.getElementById("dummy");
    feedbackElement.innerText = "Training in progress... Please wait.";
    

    const progressContainer = document.createElement('div');
    progressContainer.className = 'training-progress';
    progressContainer.style.width = '100%';
    progressContainer.style.height = '10px';
    progressContainer.style.backgroundColor = 'rgba(0, 255, 255, 0.1)';
    progressContainer.style.borderRadius = '5px';
    progressContainer.style.marginTop = '10px';
    progressContainer.style.overflow = 'hidden';
    
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    progressBar.style.height = '100%';
    progressBar.style.width = '0%';
    progressBar.style.backgroundColor = '#00ffff';
    progressBar.style.boxShadow = '0 0 10px #00ffff';
    progressBar.style.transition = 'width 0.3s ease';
    
    progressContainer.appendChild(progressBar);
    feedbackElement.appendChild(progressContainer);
    

    dataset.ys = null;
    dataset.encodeLabels(multiplayerMode || customSamples > 0 ? 6 : 5);
    

    if (!model) {
        model = tf.sequential({
            layers: [
                tf.layers.flatten({ inputShape: [7, 7, 1024] }),
                tf.layers.dense({ 
                    units: 128, 
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({l2: 0.001})
                }),
                tf.layers.dropout({ rate: 0.4 }),
                tf.layers.dense({ 
                    units: 64, 
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({l2: 0.001})
                }),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({ 
                    units: multiplayerMode || customSamples > 0 ? 6 : 5, 
                    activation: 'softmax' 
                })
            ]
        });
    }
    

    const learningRate = 0.0001;
    const optimizer = tf.train.adam(learningRate);
    
    model.compile({ 
        optimizer, 
        loss: 'categoricalCrossentropy', 
        metrics: ['accuracy'] 
    });
   

    let bestLoss = Infinity;
    let patience = 3;
    let counter = 0;
    
    try {
        const actualEpochs = boostMode ? epochs * 2 : epochs;
        const batchSize = Math.min(32, Math.floor(dataset.labels.length / 2));
        
        for (let epoch = 0; epoch < actualEpochs; epoch++) {
            const history = await model.fit(dataset.xs, dataset.ys, {
                epochs: 1,
                batchSize: batchSize,
                shuffle: true,
                callbacks: {
                    onBatchEnd: async (batch, logs) => {
                        feedbackElement.querySelector('.training-progress').style.display = 'block';
                        feedbackElement.querySelector('.progress-bar').style.width = `${((epoch + (batch+1)/dataset.xs.shape[0]) / actualEpochs) * 100}%`;
                        
                        document.getElementById("dummy").innerText = 
                            `Training: Epoch ${epoch+1}/${actualEpochs} - Loss: ${logs.loss.toFixed(5)} - Accuracy: ${(logs.acc * 100).toFixed(1)}%`;
                        
                        feedbackElement.appendChild(progressContainer);
                        

                        await tf.nextFrame();
                    }
                }
            });
            
            const loss = history.history.loss[0];
            const accuracy = history.history.acc[0];
            
            trainingStats.loss = loss.toFixed(5);
            trainingStats.accuracy = (accuracy * 100).toFixed(2);
            
            document.getElementById("modelStats").innerText = 
                `Model Stats - Accuracy: ${trainingStats.accuracy}% | Loss: ${trainingStats.loss}`;
            

            if (loss < bestLoss) {
                bestLoss = loss;
                counter = 0;
            } else {
                counter++;
                if (counter >= patience && accuracy > 0.85) {
                    console.log(`Early stopping at epoch ${epoch+1}`);
                    break;
                }
            }
            

            progressBar.style.width = `${((epoch + 1) / actualEpochs) * 100}%`;
            

            await tf.nextFrame();
        }
        

        cleanup();
        
        feedbackElement.innerText = 
            `Training complete! Model Accuracy: ${trainingStats.accuracy}%. You can now start predicting.`;
        
        updateStatus('ready');
        showLoading(false);
        return true;
    } catch (error) {
        console.error("Training failed:", error);
        feedbackElement.innerText = "Training failed. Check console for details.";
        updateStatus('error');
        showLoading(false);
        return false;
    }
}

function handleButton(elem) {
  const video = document.getElementById("wc");
  if (!video.videoWidth) {
    console.error("Video not ready - Width:", video.videoWidth);
    alert("Camera not ready—check console!");
    return;
  }
    

    elem.classList.add('glow');
    setTimeout(() => elem.classList.remove('glow'), 300);
    
  switch(elem.id) {
    case "0":
      rockSamples++;
      document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
      break;
    case "1":
      paperSamples++;
      document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
      break;
    case "2":
      scissorsSamples++;
      document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
      break;  
    case "3":
      spockSamples++;
      document.getElementById("spocksamples").innerText = "Spock samples:" + spockSamples;
      break;
    case "4":
      lizardSamples++;
      document.getElementById("lizardsamples").innerText = "Lizard samples:" + lizardSamples;
      break;
    case "5":
      customSamples++;
      document.getElementById("customsamples").innerText = "Custom samples:" + customSamples;
      break;
  }
    

    tf.tidy(() => {
  const label = parseInt(elem.id);
  const img = tf.browser.fromPixels(video).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
  dataset.addExample(mobilenet.predict(img), label);
    });
    

    const totalSamples = rockSamples + paperSamples + scissorsSamples + spockSamples + lizardSamples + customSamples;
    document.getElementById("dummy").innerHTML = `${totalSamples} total samples collected. ${totalSamples > 15 ? '<strong>Ready to train!</strong>' : 'Collect more samples...'}`;
    

    if (totalSamples >= 15 && !model) {
        document.getElementById("train").classList.add('glow');
        setTimeout(() => document.getElementById("train").classList.remove('glow'), 2000);
    }
}

function smoothPredictions(newPrediction) {

    predictionBuffer.push(newPrediction);
    if (predictionBuffer.length > 5) {
        predictionBuffer.shift();
    }
    

    const counts = {};
    predictionBuffer.forEach(p => {
        counts[p] = (counts[p] || 0) + 1;
    });
    

    let maxCount = 0;
    let mostCommon = -1;
    
    for (const [pred, count] of Object.entries(counts)) {
        if (count > maxCount) {
            maxCount = count;
            mostCommon = parseInt(pred);
        }
    }
    

    if (maxCount >= 3 || (predictionBuffer.length < 5 && maxCount >= 2)) {
        return mostCommon;
    }
    

    return lastPrediction;
}

async function predict() {
  if (!mobilenet || !model) {
    console.error("MobileNet or model not loaded yet. Please train or load first.");
        stopPredicting();
    return;
  }
    
    updateStatus('predicting');
    document.getElementById('startPredicting').classList.add('predicting');
    
  const video = document.getElementById("wc");
    const predictionElement = document.getElementById("prediction");
    
    try {
  while (isPredicting) {

            frameCount++;
            if (frameCount % 2 !== 0) {

    if (hands) await hands.send({ image: video });
                await tf.nextFrame();
                continue;
            }
            

            if (predictionThrottle) {
                await tf.nextFrame();
                continue;
            }
            
            predictionThrottle = true;
            

            const handGuideOverlay = document.querySelector('.hand-guide-overlay');
            const isHandDetected = handGuideOverlay.classList.contains('hand-detected');
            
            if (!isHandDetected) {

                predictionElement.innerText = "Place your hand in the frame";
                predictionElement.classList.remove('active');
                void predictionElement.offsetWidth; // Force reflow
                predictionElement.classList.add('active');
                

                setTimeout(() => {
                    predictionThrottle = false;
                }, 50);
                
                await tf.nextFrame();
                continue;
            }
            

    const predictedClass = tf.tidy(() => {
                const img = tf.browser.fromPixels(video)
                    .resizeNearestNeighbor([224, 224])
                    .toFloat()
                    .expandDims();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
                return { 
                    classId: predictions.as1D().argMax(), 
                    probs: predictions 
                };
    });
    
    const classId = (await predictedClass.classId.data())[0];
            const probs = await predictedClass.probs.data();
            const confidence = (Math.max(...probs) * 100).toFixed(1);
            

            const smoothedClassId = smoothPredictions(classId);
            
            if (smoothedClassId !== lastPrediction) {

                predictionElement.classList.remove('active');
                void predictionElement.offsetWidth; // Force reflow
                predictionElement.classList.add('active');
                

                lastPrediction = smoothedClassId;
            }
    
    const gestureNames = ["Rock", "Paper", "Scissors", "Spock", "Lizard", "Custom"];
            let predictionText = `I see ${gestureNames[smoothedClassId]}—${confidence}%`;
            predictionElement.innerText = predictionText;


    document.querySelectorAll(".button-group button").forEach(btn => btn.classList.remove("glow"));
            

            if (confidence > 70) {

                const gestureButton = document.getElementById(smoothedClassId);
                if (gestureButton) {
                    gestureButton.classList.add("glow");
                }
            }
            

            if (predictionHistory.length === 0 || predictionHistory[0] !== gestureNames[smoothedClassId]) {
                predictionHistory.unshift(gestureNames[smoothedClassId]);
    if (predictionHistory.length > 10) predictionHistory.pop();
    document.getElementById("history").innerText = "History: " + predictionHistory.join(", ");
            }


    if (speedChallenge) {
                handleSpeedChallenge(smoothedClassId);
    } else if (multiplayerMode) {
                handleMultiplayer(smoothedClassId);
    } else {
      const aiMove = Math.floor(Math.random() * 5);
                const result = getGameResult(smoothedClassId, aiMove);
                document.getElementById("gameResult").innerText = `You: ${gestureNames[smoothedClassId]} | AI: ${gestureNames[aiMove]} | ${result}`;
      updateScores(result);
    }

    predictedClass.classId.dispose();
    predictedClass.probs.dispose();
            

            setTimeout(() => {
                predictionThrottle = false;
            }, 50);
            

    await tf.nextFrame();
        }
    } catch (error) {
        console.error("Prediction error:", error);
        stopPredicting();
  }
}

function getGameResult(player, ai) {
  if (player === ai) return "Tie!";
  const wins = [
    [0, 2], [0, 4], [1, 0], [1, 3], [2, 1], [2, 4], [3, 2], [3, 0], [4, 1], [4, 3]
  ];
  return wins.some(([w, l]) => w === player && l === ai) ? "You Win!" : "AI Wins!";
}

function updateScores(result) {
  if (result === "You Win!") scores.wins++;
  else if (result === "AI Wins!") scores.losses++;
  else scores.ties++;
  document.getElementById("scores").innerText = `Scores - You: ${scores.wins} | AI: ${scores.losses} | Ties: ${scores.ties}`;
  

  const scoresElement = document.getElementById("scores");
  scoresElement.classList.remove('highlight-score');
  void scoresElement.offsetWidth; // Force reflow
  scoresElement.classList.add('highlight-score');
  
  updateLeaderboard();
}

function updateLeaderboard() {
  const playerName = "Player";
  const scoreEntry = { name: playerName, wins: scores.wins };
  leaderboard = leaderboard.filter(entry => entry.name !== playerName);
  leaderboard.push(scoreEntry);
  leaderboard.sort((a, b) => b.wins - a.wins);
  leaderboard = leaderboard.slice(0, 5);
  localStorage.setItem('rpslsLeaderboard', JSON.stringify(leaderboard));
  document.getElementById("leaderboard").innerText = "Leaderboard: " + leaderboard.map(e => `${e.name}: ${e.wins}`).join(" | ");
}

function handleSpeedChallenge(classId) {
  if (speedSequence.length === 0) speedSequence = [0, 1, 2, 3, 4];
  if (classId === speedSequence[0]) {
    speedSequence.shift();

        document.getElementById(classId).classList.add('glow');
        setTimeout(() => document.getElementById(classId).classList.remove('glow'), 500);
        
    if (speedSequence.length === 0) {
      const timeTaken = ((Date.now() - speedStartTime) / 1000).toFixed(2);
      const speedResult = document.getElementById("speedResult");
      speedResult.innerText = `Speed Challenge Completed in ${timeTaken}s!`;
      

      speedResult.classList.add('celebration');
      setTimeout(() => speedResult.classList.remove('celebration'), 3000);
      
      speedChallenge = false;
    } else {

      const gestureNames = ["Rock", "Paper", "Scissors", "Spock", "Lizard"];
      const remainingGestures = speedSequence.map(id => gestureNames[id]).join(", ");
      document.getElementById("speedResult").innerText = `Next: ${remainingGestures}`;
    }
  }
}

async function handleMultiplayer(classId1) {
  const video = document.getElementById("wc");
  const predictedClass2 = tf.tidy(() => {
    const img = tf.browser.fromPixels(video).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
    const activation = mobilenet.predict(img);
    const predictions = model.predict(activation);
    return predictions.as1D().argMax();
  });
  const classId2 = (await predictedClass2.data())[0];
  const gestureNames = ["Rock", "Paper", "Scissors", "Spock", "Lizard", "Custom"];
  const result = getGameResult(classId2, classId1);
  document.getElementById("gameResult").innerText = `P1: ${gestureNames[classId1]} | P2: ${gestureNames[classId2]} | ${result.replace("You Win!", "P1 Wins!").replace("AI Wins!", "P2 Wins!")}`;
  predictedClass2.dispose();
}

async function doTraining() {
    document.getElementById("train").disabled = true;
    document.getElementById("train").classList.add('btn-primary');
    const success = await train();
    document.getElementById("train").disabled = false;
    
    if (success) {

        document.getElementById("train").classList.add('glow');
        setTimeout(() => document.getElementById("train").classList.remove('glow'), 1000);

        document.getElementById("startPredicting").classList.add('glow');
        setTimeout(() => document.getElementById("startPredicting").classList.remove('glow'), 2000);
  alert("Training Done!");
    }
}

async function addMoreSamples() {
  if (!model || !dataset.xs) {
    alert("Train a model first before adding more samples!");
    return;
  }
    
    document.getElementById("moreSamples").disabled = true;
  await train(3);
    document.getElementById("moreSamples").disabled = false;
  alert("Added more samples and fine-tuned!");
}

function startPredicting() {
  if (!model) {
    alert("Please train the model or load a saved model first!");
    return;
  }
  if (!mobilenet) {
    alert("MobileNet not loaded yet—please wait a sec!");
    return;
  }
    

  isPredicting = true;
    frameCount = 0;
    predictionThrottle = false;
    predictionBuffer = [];
    lastPrediction = -1;
    

    document.getElementById("prediction").innerText = "";
    document.getElementById("gameResult").innerText = "";
    document.getElementById("history").innerText = "";
    predictionHistory = [];
    
    document.getElementById('startPredicting').classList.add('glow', 'btn-primary');
    document.getElementById('stopPredicting').classList.remove('glow');
    

    document.querySelector('.hand-guide-overlay').style.display = 'flex';
    
  predict();
}

function stopPredicting() {
  isPredicting = false;
  speedChallenge = false;
    
    document.getElementById('startPredicting').classList.remove('glow', 'predicting', 'btn-primary');
    document.getElementById('stopPredicting').classList.add('glow');
    setTimeout(() => document.getElementById('stopPredicting').classList.remove('glow'), 1000);
    

    document.querySelector('.hand-guide-overlay').classList.remove('hand-detected');
    
    updateStatus('ready');
  document.querySelectorAll(".button-group button").forEach(btn => btn.classList.remove("glow"));
    
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function saveModel() {
  if (!model) {
    alert("No model to save. Please train first!");
    return;
  }
    
    showLoading(true);
    

    Promise.all([
        model.save('downloads://my_model'),
        model.save('indexeddb://my_model')
    ])
    .then(() => {
        showLoading(false);
        document.getElementById("saveModel").classList.add('glow');
        setTimeout(() => document.getElementById("saveModel").classList.remove('glow'), 1000);
        document.getElementById("dummy").innerText = "Model saved to downloads and browser storage. You can load it anytime!";
    })
    .catch(error => {
        showLoading(false);
        console.error("Failed to save model:", error);
        alert("Couldn't save the model completely—check the console for details!");
    });
}

async function loadModel() {
  try {
        updateStatus('loading');
        showLoading(true);
        
        document.getElementById("loadModel").disabled = true;
        

        let loadedModel = null;
        
        try {

            console.log("Trying to load from IndexedDB...");
            loadedModel = await tf.loadLayersModel('indexeddb://my_model');
            console.log("Successfully loaded from IndexedDB");
        } catch (indexedDbError) {
            console.log("Failed to load from IndexedDB:", indexedDbError.message);
            
            try {

                console.log("Trying to load from server path...");
                loadedModel = await tf.loadLayersModel('my_model.json');
                console.log("Successfully loaded from server path");
            } catch (localError) {
                console.log("Failed to load from local path:", localError.message);
                
                try {

                    console.log("Trying to load from localhost...");
                    loadedModel = await tf.loadLayersModel('http://127.0.0.1:5500/my_model.json');
                    console.log("Successfully loaded from localhost");
                } catch (localhostError) {
                    console.log("Failed to load from localhost:", localhostError.message);
                    throw new Error("All loading attempts failed");
                }
            }
        }
        
        if (loadedModel) {
            model = loadedModel;
            

            try {
                await model.save('indexeddb://my_model');
                console.log("Model saved to IndexedDB for future use");
            } catch (saveError) {
                console.log("Note: Could not save model to IndexedDB:", saveError.message);
            }
            
            document.getElementById("loadModel").disabled = false;
            document.getElementById("loadModel").classList.add('glow');
            setTimeout(() => document.getElementById("loadModel").classList.remove('glow'), 1000);
            
            showLoading(false);
            updateStatus('ready');
            

            document.getElementById("dummy").innerText = "Model loaded successfully! You can now start predicting.";
            
    alert("Model loaded successfully!");
            return true;
        }
  } catch (error) {
        document.getElementById("loadModel").disabled = false;
    console.error("Failed to load model:", error);
        showLoading(false);
        updateStatus('error');
        

        createModelFileInput();
        

        let errorMsg = "Couldn't load the model automatically. You can:";
        errorMsg += "\n1. Check your network connection";
        errorMsg += "\n2. Ensure model files are in the correct location";
        errorMsg += "\n3. Try the manual file upload option that appeared";
        
        alert(errorMsg);
        return false;
    }
}

function createModelFileInput() {

    const existingInput = document.getElementById('modelFileInput');
    if (existingInput) {
        existingInput.parentNode.removeChild(existingInput);
    }
    

    const container = document.createElement('div');
    container.id = 'modelFileInput';
    container.style.margin = '10px 0';
    container.style.padding = '10px';
    container.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
    container.style.borderRadius = '8px';
    container.style.border = '1px solid rgba(255, 0, 0, 0.3)';
    

    const header = document.createElement('div');
    header.innerText = 'Manual Model Loading';
    header.style.fontWeight = 'bold';
    header.style.marginBottom = '8px';
    header.style.color = '#ff5500';
    container.appendChild(header);
    

    const description = document.createElement('div');
    description.innerText = 'Select both model.json and weights.bin files:';
    description.style.fontSize = '12px';
    description.style.marginBottom = '8px';
    container.appendChild(description);
    

    const inputJson = document.createElement('input');
    inputJson.type = 'file';
    inputJson.id = 'modelJsonInput';
    inputJson.accept = '.json';
    inputJson.style.width = '100%';
    inputJson.style.marginBottom = '8px';
    inputJson.style.fontSize = '12px';
    
    const inputLabel1 = document.createElement('label');
    inputLabel1.innerText = 'model.json:';
    inputLabel1.style.display = 'block';
    inputLabel1.style.fontSize = '12px';
    inputLabel1.style.marginBottom = '4px';
    
    container.appendChild(inputLabel1);
    container.appendChild(inputJson);
    
    const inputWeights = document.createElement('input');
    inputWeights.type = 'file';
    inputWeights.id = 'weightsInput';
    inputWeights.accept = '.bin';
    inputWeights.style.width = '100%';
    inputWeights.style.marginBottom = '8px';
    inputWeights.style.fontSize = '12px';
    
    const inputLabel2 = document.createElement('label');
    inputLabel2.innerText = 'weights.bin:';
    inputLabel2.style.display = 'block';
    inputLabel2.style.fontSize = '12px';
    inputLabel2.style.marginBottom = '4px';
    
    container.appendChild(inputLabel2);
    container.appendChild(inputWeights);
    

    const loadButton = document.createElement('button');
    loadButton.innerText = 'Load Selected Model Files';
    loadButton.classList.add('manualLoadBtn');
    loadButton.style.width = '100%';
    loadButton.style.marginTop = '8px';
    container.appendChild(loadButton);
    

    const statusText = document.createElement('div');
    statusText.id = 'modelLoadStatus';
    statusText.style.marginTop = '8px';
    statusText.style.fontSize = '12px';
    statusText.style.color = '#ccc';
    container.appendChild(statusText);
    

    const loadModelBtn = document.getElementById('loadModel');
    loadModelBtn.parentNode.insertBefore(container, loadModelBtn.nextSibling);
    

    loadButton.addEventListener('click', async () => {
        const jsonFile = document.getElementById('modelJsonInput').files[0];
        const weightsFile = document.getElementById('weightsInput').files[0];
        
        if (!jsonFile) {
            statusText.innerText = 'Please select a model.json file';
            statusText.style.color = '#ff5500';
            return;
        }
        
        if (!weightsFile) {
            statusText.innerText = 'Please select a weights.bin file';
            statusText.style.color = '#ff5500';
            return;
        }
        
        statusText.innerText = 'Loading model files...';
        statusText.style.color = '#00ffff';
        
        try {
            showLoading(true);
            updateStatus('loading');
            

            const modelUrl = URL.createObjectURL(jsonFile);
            const weightsUrl = URL.createObjectURL(weightsFile);
            

            const jsonContent = await jsonFile.text();
            const modelJSON = JSON.parse(jsonContent);
            

            const modelElement = document.createElement('div');
            modelElement.style.display = 'none';
            document.body.appendChild(modelElement);
            

            const weightsHandler = tf.io.browserFiles([weightsFile]);
            

            model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
            
            showLoading(false);
            updateStatus('ready');
            

            document.getElementById("dummy").innerText = "Model loaded successfully from file! You can now start predicting.";
            container.style.backgroundColor = 'rgba(0, 255, 0, 0.1)';
            container.style.border = '1px solid rgba(0, 255, 0, 0.3)';
            header.style.color = '#00ff00';
            header.innerText = 'Model Loaded Successfully';
            statusText.innerText = 'Model loaded successfully!';
            statusText.style.color = '#00ff00';
            

            try {
                await model.save('indexeddb://my_model');
                console.log("Model saved to IndexedDB for future use");
                statusText.innerText = 'Model loaded and saved to browser storage for future use!';
            } catch (saveError) {
                console.log("Note: Could not save model to IndexedDB:", saveError.message);
            }
            

            loadButton.classList.add('glow');
            setTimeout(() => loadButton.classList.remove('glow'), 1000);
            

            URL.revokeObjectURL(modelUrl);
            URL.revokeObjectURL(weightsUrl);
            document.body.removeChild(modelElement);
            
        } catch (error) {
            console.error("Failed to load model from files:", error);
            showLoading(false);
            updateStatus('error');
            
            statusText.innerText = 'Error: ' + error.message;
            statusText.style.color = '#ff5500';
            
            container.style.backgroundColor = 'rgba(255, 0, 0, 0.2)';
            alert("Failed to load model from files. Error: " + error.message);
        }
    });
    

    inputJson.addEventListener('change', () => {
        const file = inputJson.files[0];
        if (file) {
            if (file.name.endsWith('.json')) {
                inputLabel1.style.color = '#00ff00';
                validateFiles();
            } else {
                inputLabel1.style.color = '#ff5500';
                statusText.innerText = 'Please select a .json file';
                statusText.style.color = '#ff5500';
            }
        }
    });
    
    inputWeights.addEventListener('change', () => {
        const file = inputWeights.files[0];
        if (file) {
            if (file.name.endsWith('.bin')) {
                inputLabel2.style.color = '#00ff00';
                validateFiles();
            } else {
                inputLabel2.style.color = '#ff5500';
                statusText.innerText = 'Please select a .bin file';
                statusText.style.color = '#ff5500';
            }
        }
    });
    
    function validateFiles() {
        const jsonFile = inputJson.files[0];
        const weightsFile = inputWeights.files[0];
        
        if (jsonFile && weightsFile) {
            const jsonName = jsonFile.name.replace('.json', '');
            const weightsName = weightsFile.name.replace('.weights.bin', '').replace('.bin', '');
            
            if (jsonName === weightsName || weightsFile.name === 'my_model.weights.bin') {
                statusText.innerText = 'Files match! Ready to load.';
                statusText.style.color = '#00ff00';
                loadButton.disabled = false;
            } else {
                statusText.innerText = 'Warning: File names don\'t match, but you can still try loading.';
                statusText.style.color = '#ffff00';
                loadButton.disabled = false;
            }
        }
  }
}

function toggleTutorial() {
  const tutorial = document.getElementById("tutorial");
    if (tutorial.style.display === "block") {
        tutorial.style.opacity = "1";
        setTimeout(() => {
            tutorial.style.opacity = "0";
            setTimeout(() => {
                tutorial.style.display = "none";
            }, 300);
        }, 50);
    } else {
        tutorial.style.display = "block";
        setTimeout(() => {
            tutorial.style.opacity = "1";
        }, 50);
    }
}

function saveDataset() {
  if (!dataset.xs || !dataset.ys) {
    alert("No dataset to save. Add samples first!");
    return;
  }
    
    showLoading(true);
    try {
  const xsData = Array.from(dataset.xs.dataSync());
  const ysData = Array.from(dataset.ys.dataSync());
  const data = { xs: xsData, ys: ysData };
  const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "rpsls_dataset.json";
  a.click();
  URL.revokeObjectURL(url);
        
        showLoading(false);
        document.getElementById("saveDataset").classList.add('glow');
        setTimeout(() => document.getElementById("saveDataset").classList.remove('glow'), 1000);
    } catch (error) {
        showLoading(false);
        console.error("Failed to save dataset:", error);
        alert("Failed to save dataset. Check console for details.");
    }
}

function toggleBoostMode() {
  boostMode = !boostMode;
  document.getElementById("boostMode").innerText = `Boost Mode: ${boostMode ? "On" : "Off"}`;
    document.getElementById("boostMode").classList.add('glow');
    setTimeout(() => document.getElementById("boostMode").classList.remove('glow'), 1000);
  alert(`Boost Mode ${boostMode ? "enabled—double epochs!" : "disabled—normal training."}`);
}

function toggleMultiplayer() {
  multiplayerMode = !multiplayerMode;
  document.getElementById("multiplayer").innerText = `Multiplayer: ${multiplayerMode ? "On" : "Off"}`;
    document.getElementById("multiplayer").classList.add('glow');
    setTimeout(() => document.getElementById("multiplayer").classList.remove('glow'), 1000);
    
    if (hands) {
        hands.setOptions({ maxNumHands: multiplayerMode ? 2 : 1 });
    }
    
  alert(`Multiplayer Mode ${multiplayerMode ? "enabled—two players!" : "disabled—single player."}`);
}

function startSpeedChallenge() {
  if (!model) {
    alert("Train or load a model first!");
    return;
  }
    
  speedChallenge = true;
  speedStartTime = Date.now();
  speedSequence = [0, 1, 2, 3, 4];
  
  const speedResultElement = document.getElementById("speedResult");
  speedResultElement.innerText = "Show: Rock, Paper, Scissors, Spock, Lizard!";
  speedResultElement.classList.add('active');
  
  document.getElementById("speedChallenge").classList.add('glow');
    

    if (!isPredicting) {
        startPredicting();
    }
}

function toggleMusic() {
  const bgMusic = document.getElementById("bgMusic");
    const musicBtn = document.getElementById("music");
    
  if (bgMusic.paused) {
        bgMusic.volume = 0.5; // Set volume to 50%
        bgMusic.play().then(() => {
            musicBtn.innerText = "Music: On";
            musicBtn.classList.add('glow');
            setTimeout(() => musicBtn.classList.remove('glow'), 1000);
        }).catch(err => {
      console.error("Music playback failed:", err);
      alert("Music failed—check console!");
    });
  } else {
    bgMusic.pause();
        musicBtn.innerText = "Music: Off";
        musicBtn.classList.remove('glow');
  }
}

function onHandsResults(results) {
  if (!ctx) return;
    
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  

  const handGuideOverlay = document.querySelector('.hand-guide-overlay');
    
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {

        handGuideOverlay.classList.add('hand-detected');
        

        ctx.save();
        
    results.multiHandLandmarks.forEach((landmarks, index) => {

            ctx.strokeStyle = index === 0 ? "rgba(0, 255, 0, 0.8)" : "rgba(0, 204, 0, 0.8)";
      ctx.lineWidth = 2;
            

      const bbox = getBoundingBox(landmarks);
            

      ctx.beginPath();
            const radius = 10;
            ctx.moveTo(bbox.xMin + radius, bbox.yMin);
            ctx.lineTo(bbox.xMax - radius, bbox.yMin);
            ctx.arcTo(bbox.xMax, bbox.yMin, bbox.xMax, bbox.yMin + radius, radius);
            ctx.lineTo(bbox.xMax, bbox.yMax - radius);
            ctx.arcTo(bbox.xMax, bbox.yMax, bbox.xMax - radius, bbox.yMax, radius);
            ctx.lineTo(bbox.xMin + radius, bbox.yMax);
            ctx.arcTo(bbox.xMin, bbox.yMax, bbox.xMin, bbox.yMax - radius, radius);
            ctx.lineTo(bbox.xMin, bbox.yMin + radius);
            ctx.arcTo(bbox.xMin, bbox.yMin, bbox.xMin + radius, bbox.yMin, radius);
            ctx.stroke();
            

            ctx.shadowColor = "rgba(0, 255, 0, 0.7)";
            ctx.shadowBlur = 10;
      ctx.stroke();
            ctx.shadowBlur = 0;
            

            drawConnectors(ctx, landmarks);
            

            const keyPoints = [0, 4, 8, 12, 16, 20]; // thumb tip, finger tips
            ctx.fillStyle = "rgba(0, 255, 255, 0.8)";
            keyPoints.forEach(point => {
                if (landmarks[point]) {
                    const x = landmarks[point].x * canvas.width;
                    const y = landmarks[point].y * canvas.height;
                    ctx.beginPath();
                    ctx.arc(x, y, 5, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        });
        
        ctx.restore();
  } else {

        handGuideOverlay.classList.remove('hand-detected');
        

        ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
    ctx.lineWidth = 2;
        

        ctx.setLineDash([5, 5]);
    ctx.beginPath();
        ctx.rect(15, 15, canvas.width - 30, canvas.height - 30);
    ctx.stroke();
        ctx.setLineDash([]);
  }
}


function drawConnectors(ctx, landmarks) {

    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // index finger
        [5, 9], [9, 10], [10, 11], [11, 12], // middle finger
        [9, 13], [13, 14], [14, 15], [15, 16], // ring finger
        [13, 17], [17, 18], [18, 19], [19, 20], // pinky
        [0, 17], [5, 9], [9, 13], [13, 17] // palm
    ];
    
    ctx.strokeStyle = "rgba(0, 255, 255, 0.5)";
    ctx.lineWidth = 1.5;
    
    connections.forEach(([i, j]) => {
        if (landmarks[i] && landmarks[j]) {
            ctx.beginPath();
            ctx.moveTo(landmarks[i].x * canvas.width, landmarks[i].y * canvas.height);
            ctx.lineTo(landmarks[j].x * canvas.width, landmarks[j].y * canvas.height);
            ctx.stroke();
        }
    });
}

function getBoundingBox(landmarks) {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    
  landmarks.forEach(lm => {
    xMin = Math.min(xMin, lm.x * canvas.width);
    xMax = Math.max(xMax, lm.x * canvas.width);
    yMin = Math.min(yMin, lm.y * canvas.height);
    yMax = Math.max(yMax, lm.y * canvas.height);
  });
    

    const padding = 10;
    xMin = Math.max(0, xMin - padding);
    xMax = Math.min(canvas.width, xMax + padding);
    yMin = Math.max(0, yMin - padding);
    yMax = Math.min(canvas.height, yMax + padding);
    
  return { xMin, xMax, yMin, yMax };
}


async function preloadModels() {
    try {
        // First try to load the pre-trained model from the server
        try {
            console.log("Trying to load pre-trained model...");
            model = await tf.loadLayersModel('my_model.json');
            console.log("Successfully loaded pre-trained model");
            document.getElementById("dummy").innerText = "Pre-trained model loaded! Ready to make predictions.";
            
            // Save to IndexedDB for future use
            try {
                await model.save('indexeddb://my_model');
                console.log("Pre-trained model saved to IndexedDB for future use");
            } catch (e) {
                console.log("Could not save pre-trained model to IndexedDB:", e.message);
            }
            
            return true;
        } catch (e) {
            console.log("Failed to load pre-trained model:", e.message);
            
            // Fall back to IndexedDB
            try {
                model = await tf.loadLayersModel('indexeddb://my_model');
                console.log("Preloaded model from IndexedDB");
                document.getElementById("dummy").innerText = "Saved model found! Ready to make predictions.";
                return true;
            } catch (e) {
                console.log("No saved model in IndexedDB:", e.message);
            }
        }
        
        return false;
    } catch (error) {
        console.error("Error in preload:", error);
        return false;
    }
}


async function setupWebcam(constraints) {
    try {

        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
        }
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
  const video = document.getElementById("wc");
        
        video.srcObject = stream;
        videoStream = stream;
        
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play()
                    .then(() => {
                        console.log("Video playing - Ready:", !video.paused);
                        resolve(true);
                    })
                    .catch(err => {
                        console.error("Video play failed:", err);
                        resolve(false);
                    });
            };
            
            video.onerror = () => {
                resolve(false);
            };
        });
    } catch (error) {
        console.error("Webcam setup failed:", error);
        return false;
    }
}


async function tryWebcamWithFallbacks() {
    const video = document.getElementById("wc");
    showLoading(true, "Setting up camera...");
    

    const highQuality = { 
        video: { 
            width: { ideal: 224 },
            height: { ideal: 224 },
            facingMode: "user",
            frameRate: { ideal: 30 }
        }
    };
    

    const mediumQuality = {
        video: {
            width: { ideal: 224 },
            height: { ideal: 224 },
            facingMode: "user",
            frameRate: { ideal: 15 }
        }
    };
    

    const lowQuality = {
        video: {
            facingMode: "user"
        }
    };
    

    let success = await setupWebcam(highQuality);
    if (!success) {
        console.log("High quality webcam failed, trying medium quality...");
        document.getElementById("dummy").innerText = "Trying alternative camera settings...";
        success = await setupWebcam(mediumQuality);
    }
    
    if (!success) {
        console.log("Medium quality webcam failed, trying low quality...");
        document.getElementById("dummy").innerText = "Trying minimal camera settings...";
        success = await setupWebcam(lowQuality);
    }
    
    if (success) {
        document.getElementById("dummy").innerText = "Webcam ready—Make gestures and collect samples!";
        return true;
    } else {
        document.getElementById("dummy").innerText = "Camera setup failed. Check browser permissions.";

        const container = document.createElement('div');
        container.style.margin = '10px 0';
        container.style.padding = '10px';
        container.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
        container.style.borderRadius = '8px';
        container.style.fontSize = '14px';
        container.style.textAlign = 'left';
        container.innerHTML = `
            <p><strong>Camera access denied or unavailable. Try these steps:</strong></p>
            <ol style="padding-left: 20px; margin-top: 5px;">
                <li>Check that your camera is connected</li>
                <li>Allow camera access in your browser settings</li>
                <li>Try a different browser (Chrome recommended)</li>
                <li>Reload the page and try again</li>
            </ol>
        `;
        

        const dummyElement = document.getElementById("dummy");
        dummyElement.parentNode.insertBefore(container, dummyElement.nextSibling);
        
        return false;
    }
}

// Function to guide user to start playing directly with pre-trained model
function startWithPretrainedModel() {
    // Show a guidance message
    const feedbackElement = document.getElementById("dummy");
    feedbackElement.innerHTML = "<strong>Pre-trained model loaded!</strong> You can start playing immediately or train your own model for best results.";
    
    // Add a slight delay before showing the success indicator
    setTimeout(() => {
        document.getElementById('success-train').style.display = 'block';
        
        // Highlight the Start Game button to guide users
        document.getElementById("startPredicting").classList.add('glow');
        
        // Auto-switch to Play phase after a moment
        setTimeout(() => {
            switchToPhase('play');
            markPhaseComplete('train');
            markPhaseComplete('practice');
        }, 1500);
    }, 1000);
}

// Enhance init function with better loading sequence
async function init() {
    showLoading(true, "Initializing...");
    
    try {
        // Create dynamic loading progress updates
        const stages = [
            { name: "Loading MobileNet model...", weight: 40 },
            { name: "Setting up camera...", weight: 20 },
            { name: "Initializing hand detection...", weight: 30 },
            { name: "Loading pre-trained model...", weight: 10 }
        ];
        
        let completedWeight = 0;
        const updateProgressStage = (stageName) => {
            const stage = stages.find(s => s.name === stageName);
            if (stage) {
                completedWeight += stage.weight;
                showLoading(true, `${stageName} (${completedWeight}%)`);
            }
        };
        
        // Load MobileNet first for faster perceived performance
        updateProgressStage("Loading MobileNet model...");
        mobilenet = await loadMobilenet();
        if (!mobilenet) {
            throw new Error("Failed to load MobileNet model");
        }
        console.log("MobileNet loaded");

        // Try to set up webcam with fallbacks
        updateProgressStage("Setting up camera...");
        const webcamSuccess = await tryWebcamWithFallbacks();
        if (!webcamSuccess) {
            console.error("All webcam attempts failed");
        }
        
        // Initialize hand detection
        updateProgressStage("Initializing hand detection...");
        canvas = document.getElementById("handCanvas");
        ctx = canvas.getContext("2d");
        
        // Initialize hand guide overlay
        const handGuideOverlay = document.querySelector('.hand-guide-overlay');
        
        hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`
        });
        
        hands.setOptions({
            maxNumHands: multiplayerMode ? 2 : 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.6,
            minTrackingConfidence: 0.5
        });
        
        hands.onResults(onHandsResults);
        
        await hands.initialize();
        console.log("MediaPipe Hands initialized");
        
        // Try to preload pre-trained model
        updateProgressStage("Loading pre-trained model...");
        const modelPreloaded = await preloadModels();
        if (modelPreloaded) {
            // Highlight relevant UI elements to show ready state
            document.getElementById("loadModel").classList.add('glow');
            setTimeout(() => document.getElementById("loadModel").classList.remove('glow'), 1000);
            
            // Use the new function to guide users to play directly
            startWithPretrainedModel();
        }
        
        // Everything is loaded
        showLoading(false);
        updateStatus('ready');
        
    } catch (error) {
        console.error("Init failed:", error);
        document.getElementById("dummy").innerText = "Initialization failed: " + error.message;
        showLoading(false);
        updateStatus('error');
        
        // Show error details
        alert("Setup failed: " + error.message + "\nCheck console for details.");
    }
}


document.addEventListener('DOMContentLoaded', () => {

    loadingElement = document.getElementById('loading');
    statusElement = document.getElementById('status');
    updateStatus('initializing');
    

    init();
    

    document.querySelectorAll('.tutorial-nav-item, .tutorial-button, .tutorial-close').forEach(el => {
        el.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                el.click();
            }
        });
        el.setAttribute('tabindex', '0');
    });
});