import React from 'react';

export default function StatusPanel({ progress, statusMessage, status, onDownload }) {
  if (status === 'idle' || !status) return null;

  const isFailed = status === 'failed';
  const isDone = status === 'done';

  return (
    <div className="card border-light-subtle bg-white shadow-sm p-4 mb-4">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="fw-bold mb-0 text-dark">Processing Status</h5>
        {isDone && (
          <button onClick={onDownload} className="btn btn-success d-flex align-items-center fw-bold shadow-sm">
            <i className="bi bi-file-earmark-zip-fill me-2"></i>
            Download ZIP
          </button>
        )}
      </div>

      <div className="progress mb-3" style={{ height: '12px' }}>
        <div
          className={`progress-bar progress-bar-striped progress-bar-animated ${
            isFailed ? 'bg-danger' : isDone ? 'bg-success' : 'bg-primary'
          }`}
          role="progressbar"
          style={{ width: `${progress}%` }}
          aria-valuenow={progress}
          aria-valuemin="0"
          aria-valuemax="100"
        ></div>
      </div>

      <div className="d-flex justify-content-between align-items-center">
        <span className={`fw-semibold ${isFailed ? 'text-danger' : 'text-dark'}`}>
          {isFailed && <i className="bi bi-exclamation-triangle-fill me-2"></i>}
          {statusMessage}
        </span>
        <span className="text-secondary fw-bold">{progress}%</span>
      </div>
    </div>
  );
}
