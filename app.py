# app.py - Web API for Render
from flask import Flask, request, jsonify, send_file
from rahl import RAHLPipeline
import os
import uuid
import threading
import time

app = Flask(__name__)

# Initialize pipeline (loads model once)
print("🚀 Loading RAHL model...")
pipeline = RAHLPipeline()
print("✅ Model loaded!")

# Store job status
jobs = {}

@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>RAHL Video Generator</title>
            <style>
                body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
                input, textarea { width: 100%; padding: 10px; margin: 10px 0; }
                button { background: #4CAF50; color: white; padding: 15px 32px; border: none; cursor: pointer; }
                #result { margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>🎥 RAHL Video Generator</h1>
            <form id="genForm">
                <label>Prompt:</label>
                <textarea id="prompt" rows="3" required>a beautiful sunset over mountains</textarea>
                
                <label>Negative Prompt (optional):</label>
                <textarea id="negative" rows="2">blurry, low quality</textarea>
                
                <label>Number of Frames:</label>
                <input type="number" id="frames" value="16" min="8" max="32">
                
                <button type="submit">Generate Video</button>
            </form>
            
            <div id="result"></div>
            
            <script>
                document.getElementById('genForm').onsubmit = async (e) => {
                    e.preventDefault();
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = '⏳ Generating video... (this takes 1-2 minutes)';
                    
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            prompt: document.getElementById('prompt').value,
                            negative: document.getElementById('negative').value,
                            num_frames: parseInt(document.getElementById('frames').value)
                        })
                    });
                    
                    const data = await response.json();
                    if (data.job_id) {
                        checkStatus(data.job_id);
                    }
                };
                
                async function checkStatus(jobId) {
                    const resultDiv = document.getElementById('result');
                    const interval = setInterval(async () => {
                        const response = await fetch('/status/' + jobId);
                        const data = await response.json();
                        
                        if (data.status === 'completed') {
                            clearInterval(interval);
                            resultDiv.innerHTML = `<a href="/download/${jobId}" target="_blank">📥 Download Video</a>`;
                        } else if (data.status === 'processing') {
                            resultDiv.innerHTML = '⏳ ' + data.message;
                        }
                    }, 2000);
                }
            </script>
        </body>
    </html>
    """

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    job_id = str(uuid.uuid4())
    
    # Start generation in background
    def generate_video():
        jobs[job_id] = {'status': 'processing', 'message': 'Starting...'}
        try:
            video = pipeline.generate(
                prompt=data['prompt'],
                negative_prompt=data.get('negative', ''),
                num_frames=int(data.get('num_frames', 16))
            )
            
            # Save video
            output_path = f"/tmp/{job_id}.mp4"
            pipeline.save_video(video, output_path)
            
            jobs[job_id] = {
                'status': 'completed',
                'path': output_path
            }
        except Exception as e:
            jobs[job_id] = {'status': 'failed', 'error': str(e)}
    
    thread = threading.Thread(target=generate_video)
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    return jsonify(jobs.get(job_id, {'status': 'not_found'}))

@app.route('/download/<job_id>')
def download(job_id):
    if job_id in jobs and jobs[job_id]['status'] == 'completed':
        return send_file(jobs[job_id]['path'], as_attachment=True)
    return 'Job not found or not completed', 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
