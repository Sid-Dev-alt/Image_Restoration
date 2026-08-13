import React, { useState, useEffect } from 'react';

export default function CompareModal({ filename, sessionId, onClose }) {
  const [sliderPosition, setSliderPosition] = useState(50);

  if (!filename) return null;

  const originalUrl = `/api/image/original/${sessionId}/${filename}`;
  const restoredUrl = `/api/image/restored/${sessionId}/${filename}`;

  return (
    <div className="modal show d-block" tabIndex="-1" style={{ backgroundColor: 'rgba(0, 0, 0, 0.75)', zIndex: 1055 }}>
      <div className="modal-dialog modal-dialog-centered modal-lg">
        <div className="modal-content border-light-subtle shadow-lg">
          <div className="modal-header border-bottom border-light-subtle bg-light">
            <h5 className="modal-title fw-bold text-dark">Before / After Comparison</h5>
            <button type="button" className="btn-close" aria-label="Close" onClick={onClose}></button>
          </div>
          <div className="modal-body p-4 bg-white">
            <p className="text-muted mb-3" style={{ fontSize: '0.9rem' }}>
              File: <strong>{filename}</strong> <span className="mx-2">|</span> 
              Drag the slider to compare original (left) vs restored (right).
            </p>
            
            <div className="position-relative overflow-hidden rounded shadow-sm bg-black" style={{ width: '100%', height: '480px' }}>
              {/* Original (Base Image) */}
              <img
                src={originalUrl}
                alt="Original Blurry"
                className="position-absolute start-0 top-0 w-100 h-100"
                style={{ objectFit: 'contain', userSelect: 'none', pointerEvents: 'none' }}
              />
              
              {/* Restored (Overlay Image) */}
              <div 
                className="position-absolute start-0 top-0 h-100 border-end border-white border-2" 
                style={{ 
                  width: `${sliderPosition}%`, 
                  overflow: 'hidden', 
                  zIndex: 2,
                  boxShadow: '0 0 10px rgba(0,0,0,0.5)'
                }}
              >
                <img
                  src={restoredUrl}
                  alt="Restored Clean"
                  className="position-absolute start-0 top-0 h-100"
                  style={{ 
                    width: '752px', // matching parent max width (approx)
                    maxWidth: 'none',
                    objectFit: 'cover',
                    userSelect: 'none',
                    pointerEvents: 'none'
                  }}
                />
              </div>

              {/* Slider Handle Circle */}
              <div 
                className="position-absolute top-50 translate-middle rounded-circle bg-white shadow-sm border border-secondary d-flex align-items-center justify-content-center"
                style={{
                  left: `${sliderPosition}%`,
                  width: '36px',
                  height: '36px',
                  cursor: 'ew-resize',
                  zIndex: 10,
                  transform: 'translate(-50%, -50%)',
                  boxShadow: '0 4px 10px rgba(0, 0, 0, 0.4)'
                }}
              >
                <span className="fw-bold text-dark text-nowrap" style={{ fontSize: '0.85rem' }}>↔</span>
              </div>

              {/* Transparent Range Input Overlay */}
              <input
                type="range"
                min="0"
                max="100"
                value={sliderPosition}
                onChange={(e) => setSliderPosition(Number(e.target.value))}
                className="position-absolute start-0 top-0 w-100 h-100 opacity-0"
                style={{
                  cursor: 'ew-resize',
                  zIndex: 15,
                  WebkitAppearance: 'none',
                  appearance: 'none'
                }}
              />
            </div>
          </div>
          <div className="modal-footer border-top border-light-subtle bg-light">
            <button type="button" className="btn btn-secondary fw-semibold" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
