import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import StatusPanel from './components/StatusPanel';
import ResultsGallery from './components/ResultsGallery';
import CompareModal from './components/CompareModal';

export default function App() {
  const [files, setFiles] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  const [statusMessage, setStatusMessage] = useState('Ready to process...');
  const [images, setImages] = useState([]);
  const [compareFilename, setCompareFilename] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const fileInputRef = useRef(null);
  const pollingRef = useRef(null);

  // Clear interval on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const handleFilesSelected = (e) => {
    if (!e.target.files) return;
    const selectedFiles = Array.from(e.target.files).filter(file => {
      return file.name.match(/\.(jpg|jpeg|png|bmp)$/i);
    });

    if (selectedFiles.length === 0) {
      alert('No valid images (JPG, PNG, BMP) found in the selected folder.');
      return;
    }

    setFiles(selectedFiles);
    // Auto-trigger restoration immediately
    runRestorationPipeline(selectedFiles);
  };

  const runRestorationPipeline = async (filesToProcess) => {
    setIsProcessing(true);
    setStatus('processing');
    setProgress(0);
    setStatusMessage('Uploading images...');
    setImages([]);
    setSessionId(null);

    try {
      // Step 1: Upload images
      const formData = new FormData();
      filesToProcess.forEach((file) => {
        formData.append('files', file);
      });

      const uploadRes = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!uploadRes.ok) throw new Error('Failed to upload files.');
      const uploadData = await uploadRes.json();
      const newSessionId = uploadData.session_id;
      setSessionId(newSessionId);

      // Step 2: Trigger processing with "auto" model configuration
      setStatusMessage('Initializing AI Restoration model...');
      const processRes = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: newSessionId,
          model: 'auto'
        })
      });

      if (!processRes.ok) throw new Error('Failed to start restoration pipeline.');

      // Step 3: Start polling
      startPolling(newSessionId);

    } catch (err) {
      console.error(err);
      setStatus('failed');
      setStatusMessage(`Error: ${err.message}`);
      setIsProcessing(false);
    }
  };

  const startPolling = (sid) => {
    if (pollingRef.current) clearInterval(pollingRef.current);

    pollingRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${sid}`);
        if (!res.ok) throw new Error('Error fetching status.');
        const data = await res.json();

        const total = data.total_images;
        const processed = data.processed_images;
        const percent = total > 0 ? Math.round((processed / total) * 100) : 0;

        setProgress(percent);
        setImages(data.images);
        setSessionId(sid); // Keep state synced

        if (data.status === 'processing') {
          setStatus('processing');
          setStatusMessage(`Restoring image ${processed}/${total} using AI...`);
        } else if (data.status === 'done') {
          clearInterval(pollingRef.current);
          setStatus('done');
          setProgress(100);
          setStatusMessage('Restoration Complete! All images cleansed.');
          setIsProcessing(false);
        } else if (data.status === 'failed') {
          clearInterval(pollingRef.current);
          setStatus('failed');
          setStatusMessage('Pipeline processing failed.');
          setIsProcessing(false);
        }
      } catch (err) {
        console.error(err);
        clearInterval(pollingRef.current);
        setStatus('failed');
        setStatusMessage('Failed to connect to status updates.');
        setIsProcessing(false);
      }
    }, 1000);
  };

  const handleDownload = () => {
    if (!sessionId) return;
    window.location.href = `/api/download/${sessionId}`;
  };

  return (
    <div className="bg-light min-h-100vh text-dark pb-5" style={{ minHeight: '100vh' }}>
      <Header />
      
      <div className="container">
        <div className="row g-4">
          {/* Left Panel: Folder Selection Button */}
          <div className="col-12 col-lg-5">
            <div className="card border-light-subtle bg-white shadow-sm p-4 text-center">
              <div className="text-primary mb-3">
                <i className="bi bi-folder-fill" style={{ fontSize: '3.5rem' }}></i>
              </div>
              <h5 className="fw-bold text-dark mb-2">Restore Image Folder</h5>
              <p className="text-muted mb-4" style={{ fontSize: '0.9rem', lineHeight: '1.4' }}>
                Select an image folder to automatically restore and enhance motion blur, defocus blur, camera noise, and low-light degradation using state-of-the-art Deep Learning models.
              </p>

              <button
                onClick={() => fileInputRef.current.click()}
                disabled={isProcessing}
                className="btn btn-primary btn-lg w-100 py-3 fw-bold shadow-sm d-flex align-items-center justify-content-center"
              >
                {isProcessing ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    <span>Processing Images...</span>
                  </>
                ) : (
                  <>
                    <i className="bi bi-plus-circle-fill me-2"></i>
                    <span>Add Folder</span>
                  </>
                )}
              </button>

              <input
                ref={fileInputRef}
                type="file"
                multiple
                webkitdirectory=""
                directory=""
                style={{ display: 'none' }}
                onChange={handleFilesSelected}
              />

              {files.length > 0 && (
                <div className="alert alert-success border-success-subtle mt-4 py-2 px-3 d-flex align-items-center justify-content-center text-start">
                  <i className="bi bi-check-circle-fill me-2 fs-5 text-success"></i>
                  <div>
                    Loaded <strong className="text-success">{files.length}</strong> images for processing.
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel: Progress & Outputs */}
          <div className="col-12 col-lg-7">
            <StatusPanel 
              progress={progress} 
              statusMessage={statusMessage} 
              status={status} 
              onDownload={handleDownload} 
            />

            <ResultsGallery 
              images={images} 
              sessionId={sessionId} 
              onCompare={setCompareFilename} 
            />
          </div>
        </div>
      </div>

      {/* Compare Slider Modal */}
      {compareFilename && (
        <CompareModal
          filename={compareFilename}
          sessionId={sessionId}
          onClose={() => setCompareFilename(null)}
        />
      )}
    </div>
  );
}
